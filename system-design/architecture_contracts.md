# Architecture Contracts — Developer Reference

Detailed component interfaces, adapters, and data models. For the high-level overview, see [architecture_overview.md](architecture_overview.md).

---

## Pipeline Modules

| Module | When | Triggered by | Interface |
|---|---|---|---|
| **IndexingPipeline** | Offline (CPU default, GPU optional) | `scripts/index_book.py` | `index_source(actual_path, source_path) → IndexResult` |
| **QueryPipeline** | Online (per-request) | FastAPI `/ask`, `/search`, `/chat` | `ask(question, filters, *, history) → PipelineResult` / `ask_stream(question, *, history, filters) → Iterator[dict]` / `search(question, top_k, filters) → RetrievalResult` |

Shared adapters (Embedder, VectorStore) injected into both.

```mermaid
graph TB
  subgraph IndexingPipeline["IndexingPipeline (offline)"]
    direction TB
    IP_Parse[Parser]
    IP_Chunk[ChunkingStrategy]
    IP_Embed[Embedder]
    IP_Store[VectorStore]
    IP_Parse --> IP_Chunk --> IP_Embed --> IP_Store
  end

  subgraph QueryPipeline["QueryPipeline (online)"]
    direction TB
    QP_RS[RetrievalStrategy]
    QP_CB[ContextBuilder]
    QP_LLM[LLMClient]
    QP_Cite[CitationBuilder]
    QP_RS --> QP_CB --> QP_LLM --> QP_Cite
  end

  CLI[CLI scripts] --> IndexingPipeline
  API[FastAPI /ask /search /chat] --> QueryPipeline

  style IndexingPipeline fill:#d5f5e3
  style QueryPipeline fill:#d6eaf8
  style CLI fill:#f9f9f9
  style API fill:#f9f9f9
```

---

## Retrieval Flow

```mermaid
graph LR
  Q["retrieve(question, top_k, filters)"]

  subgraph DenseStrategy
    D_Embed[Embed query]
    D_Dense[Dense vector<br>top_k]
    D_Rerank[Reranker<br>NoOp passthrough]

    D_Embed --> D_Dense --> D_Rerank
  end

  subgraph HybridStrategy
    Embed[Embed query]
    Dense[Dense vector<br>top 30]
    Sparse[Sparse / BM25<br>top 30<br>uses raw text]
    Merge[Merge +<br>deduplicate]
    Filter[Metadata filters<br>chapter, source_type,<br>content_type]
    Rerank[Reranker<br>top 5-8]

    Embed --> Dense
    Q --> Sparse
    Dense --> Merge
    Sparse --> Merge
    Merge --> Filter
    Filter --> Rerank
  end

  subgraph "ParentChildStrategy (decorator)"
    Expand["Expand children → parents<br>via VectorStore.get_by_ids()"]
  end

  Q --> D_Embed
  D_Rerank --> Out["RetrievalResult"]
  Rerank -.->|"hybrid: direct"| Out
  Rerank -.->|"parent_child only"| Expand
  Expand -.-> Out
```

`RETRIEVAL_STRATEGY` env var controls which strategy is instantiated:
- `dense`: `DenseStrategy(embedder, store, reranker)` — embed query → dense vector search → rerank
- `hybrid` (default): `HybridStrategy(embedder, store, reranker)` — dense + sparse/BM25, merged via Reciprocal Rank Fusion

`/search` calls `QueryPipeline.search()` which delegates to `RetrievalStrategy.retrieve()`. `/ask` calls `QueryPipeline.ask()` which pipes retrieval results through ContextBuilder → LLMClient → CitationBuilder.

---

## Component Contracts

### Pipeline Modules

```mermaid
classDiagram
  class IndexingPipeline {
    +index_source(actual_path, source_path) IndexResult
  }

  class QueryPipeline {
    +ask(question, filters, *, history) PipelineResult
    +ask_stream(question, *, history, filters) Iterator~dict~
    +search(question, top_k, filters) RetrievalResult
  }

  class RetrievalStrategy {
    +retrieve(question, top_k, filters) RetrievalResult
  }

  IndexingPipeline *-- Parser
  IndexingPipeline *-- ChunkingStrategy
  IndexingPipeline *-- Embedder
  IndexingPipeline *-- VectorStore

  QueryPipeline *-- RetrievalStrategy
  QueryPipeline *-- ContextBuilder
  QueryPipeline *-- LLMClient
  QueryPipeline *-- CitationBuilder

  RetrievalStrategy *-- Embedder
  RetrievalStrategy *-- VectorStore
  RetrievalStrategy *-- Reranker
```

### Leaf Adapters — Storage & Retrieval

```mermaid
classDiagram
  class VectorStore {
    +upsert(chunks, vectors) None
    +search_dense(vector, top_k, filters) list~ScoredChunk~
    +search_sparse(text, top_k, filters) list~ScoredChunk~
    +get_by_ids(ids) list~Chunk~
    +delete_by_source(source_path) None
    +count() int
  }

  class Embedder {
    +embed_documents(texts) list~list~float~~
    +embed_query(text) list~float~
  }

  class Reranker {
    +rerank(query, candidates, top_k) list~ScoredChunk~
  }

  class RetrievalStrategy {
    +retrieve(question, top_k, filters) RetrievalResult
  }

  VectorStore <|-- QdrantStore
  Embedder <|-- LocalEmbeddingModel
  Reranker <|-- NoOpReranker
  Reranker <|-- LocalCrossEncoderReranker
  RetrievalStrategy <|-- DenseStrategy
  RetrievalStrategy <|-- HybridStrategy

  ParentChildStrategy *-- RetrievalStrategy : inner
  ParentChildStrategy *-- VectorStore : store
```

`ParentChildStrategy` is a decorator — wraps any `RetrievalStrategy` and uses `VectorStore.get_by_ids()` to expand child results to parents.

`VectorStore.upsert(chunks, vectors)` takes paired chunks and their embedding vectors. The `IndexingPipeline` calls `Embedder.embed_documents()` first, then passes both to `upsert()`.

### Leaf Adapters — Ingestion & Generation

```mermaid
classDiagram
  class Parser {
    +parse(actual_path, source_path) Document
  }

  class ChunkingStrategy {
    +chunk(document) list~Chunk~
  }

  class LLMClient {
    +chat(messages, temperature, max_tokens) GenerationResult
    +chat_stream(messages, temperature, max_tokens) StreamingGenerationResult
  }

  class ContextBuilder {
    +build(question, chunks, history) list~dict~
  }

  class CitationBuilder {
    +build(chunks) list~Citation~
  }

  Parser <|-- MarkdownParser
  Parser <|-- NotebookParser
  Parser <|-- BibParser
  Parser <|-- PdfParser
  Parser <|-- TranscriptParser
  ChunkingStrategy <|-- SectionChunker
  ChunkingStrategy <|-- NotebookChunker
  ChunkingStrategy <|-- BibChunker
  ChunkingStrategy <|-- PdfChunker
  ChunkingStrategy <|-- VideoChunker
  LLMClient <|-- OpenAICompatibleClient
```

---

## Core Data Types

### Chunk

```
Chunk (Pydantic BaseModel, frozen):
  id: str                                    # UUID
  content: str
  content_hash: str                          # dedup / change detection
  source_type: "book_text" | "video_transcript"
  content_type: str                          # concept, code_cell, figure, reference, video_segment, paper, chapter_summary
  metadata: dict[str, Any]                   # chapter, section, heading_path, url, source_path, commit_sha, etc.
  chunk_type: "parent" | "child" | "standalone"
  parent_id: str | None                      # if chunk_type == "child"
```

### ScoredChunk

```
ScoredChunk (Pydantic BaseModel):
  chunk: Chunk                               # composition, not inheritance
  score: float
  ranking_method: str                        # "dense", "sparse", "reranked"
```

### Document

```
Document (Pydantic BaseModel):
  title: str
  source_path: str
  content: str
  metadata: dict[str, Any]
  source_type: "book_text" | "video_transcript"
```

### Answer

```
Answer:
  answer: str                                # LLM-generated response
  sources: list[ScoredChunk]
  citations: list[Citation]
```

### RetrievalResult

```
RetrievalResult:
  chunks: list[ScoredChunk]
  trace: RetrievalTrace                      # per-stage trace of how the chunks were produced
```

### RetrievalTrace

```
RetrievalTrace:
  stages: list[StageTrace]
  retrieval_query: str
  total_latency_ms: int
```

### StageTrace

```
StageTrace:
  stage: str                                 # e.g. "dense", "sparse", "fusion", "reranking"
  config: dict
  latency_ms: int
  input_count: int
  output_count: int
  chunk_scores: list[ChunkStageScore]
```

### GenerationResult

```
GenerationResult:
  text: str                                  # LLM response
  prompt_tokens, completion_tokens, finish_reason, model, reasoning_content, reasoning_tokens
```

### PipelineResult

```
PipelineResult:
  answer: Answer
  retrieval_trace, generation, context_messages, system_prompt_id
```

### Citation

```
Citation:
  chunk_id: str
  source_path: str
  chapter: str
  section: str
  url: str                                   # book HTML URL (empty for non-renderable sources)
```

### IndexResult

```
IndexResult:
  source_path: str
  status: "success" | "skipped" | "failed"
  chunks_indexed: int
  error: str | None
  embeddings_model: str
  timestamp: datetime
```

### Pipelines

```
Pipelines:
  indexing: IndexingPipeline | None
  query: QueryPipeline | None
  vector_store: VectorStore | None
```

### API Request/Response Shapes

```
# POST /ask
Request:  { question: str, filters?: { chapter?: str, source_type?: str, content_type?: str } }
Response: PipelineResult

# POST /search
Request:  { question: str, top_k?: int, filters?: { chapter?: str, source_type?: str, content_type?: str } }
Response: { chunks: list[ScoredChunk] }  # from RetrievalResult.chunks

# POST /chat (SSE streaming)
Request:  { question: str, history?: list[{role: str, content: str}], filters?: {...} }
Response: SSE stream of {type: "meta"|"token"|"error"|"done", ...}

# GET /health
Response: { status: str, version: str, generation: str, chat: str }
```

---

## Adapter Wiring Contract

`api/dependencies.py` is the single source of truth for all adapter wiring. Three patterns:

| Pattern | How it works | Which seams |
|---|---|---|
| **Provider** | Env var → factory method → adapter instance | VectorStore, Embedder, LLMClient, Reranker, RetrievalStrategy |
| **Registry** | File extension → adapter map (see below) | Parser, ChunkingStrategy |
| **Direct** | One adapter, no indirection | ContextBuilder, CitationBuilder |

Two entry points:

```python
def create_indexing_pipeline(config: Settings) -> IndexingPipeline:
    """CLI (scripts/index_book.py). No retrieval wiring, create_if_missing=True."""


def create_pipelines(config: Settings) -> Pipelines:
    """FastAPI startup. Query side only, create_if_missing=False."""
```

### Registry Mappings

Parser registry (by file extension):

| Key | Adapter |
|---|---|
| `.md`, `.qmd` | MarkdownParser |
| `.ipynb` | NotebookParser |
| `.bib` | BibParser |
| `.pdf` | PdfParser |
| `.json` | TranscriptParser |

ChunkingStrategy registry (same keys as parser):

| Key | Adapter |
|---|---|
| `.md`, `.qmd` | SectionChunker |
| `.ipynb` | NotebookChunker |
| `.bib` | BibChunker |
| `.pdf` | PdfChunker |
| `.json` | VideoChunker |

Unknown keys raise `ValueError` listing supported types.

---

## Configuration Contract

Pydantic `BaseSettings`, reads from environment variables and `.env` files.

| Env var | Controls | Default |
|---|---|---|
| `EMBEDDING_PROVIDER` | Embedder factory | `"local"` |
| `EMBEDDING_MODEL_NAME` | Model name | `"BAAI/bge-large-en-v1.5"` |
| `HF_HOME` | Model cache path | `"/models"` |
| `VECTOR_STORE_PROVIDER` | VectorStore factory | `"qdrant"` |
| `QDRANT_URL` | Qdrant endpoint | `"http://qdrant:6333"` |
| `QDRANT_COLLECTION` | Collection name | `"earthrise_book"` |
| `RETRIEVAL_STRATEGY` | Strategy factory | `"hybrid"` |
| `RETRIEVAL_TOP_K` | Default result count | `8` |
| `RERANKER_PROVIDER` | Reranker factory | `"noop"` |
| `LLM_PROVIDER` | LLMClient factory | `"openai_compatible"` |
| `LLM_BASE_URL` | LLM endpoint | `"https://proxy.fast.luna.nasa.gov/v1"` |
| `LLM_API_KEY` | Auth (`SecretStr`) | `""` |
| `LLM_MODEL` | Model name | `""` |
| `LLM_TIMEOUT_SECONDS` | LLM request timeout | `60.0` |
| `APP_ENV` | Environment | `"development"` |

`.env` files are gitignored. Example config files are committed as templates.

---

## Container & Volume Contract

### Containers

| Container | Role | Online? |
|---|---|---|
| **app** | FastAPI: serves _book/ (Quarto HTML) + RAG API | Yes |
| **qdrant** | Vector DB | Yes |
| **quarto-builder** | git pull → inject widget → quarto render → _book/ | No (offline) |
| **indexer** | parse → chunk → embed → upsert to Qdrant | No (offline) |

Managed via `docker-compose.yml` at repo root. Offline containers use `profiles: ["build"]`.

### Volumes

| Volume | Purpose | Used by |
|---|---|---|
| `book_source` | Git clone of book repo (persists across pulls) | quarto-builder, indexer |
| `book_html` | Rendered _book/ output | quarto-builder (writes), app (reads) |
| `qdrant_data` | Vector storage | qdrant |
| `models_cache` | HuggingFace model weights (~1-4GB) | app, indexer |
| `data/` | Video chapter maps, transcripts | indexer |
| `logs/` | Indexing and transcription logs | indexer |

### Container Environment Overrides

The `app` service overrides these env vars via `docker-compose.yml` `environment:` block (takes precedence over `.env`):

| Var | Override value | Why |
|---|---|---|
| `QDRANT_URL` | `http://qdrant:6333` | Targets the compose service name, not localhost |
| `LLM_BASE_URL` | `http://host.docker.internal:11434/v1` | Reaches Ollama on the host from inside the container |

---

## Same-Origin Routing Contract

FastAPI serves both the API and the book HTML. Route registration order matters:

| Route | Type | Handler | Registered |
|---|---|---|---|
| `/health` | API | GET → status check | First |
| `/search` | API | POST → QueryPipeline.search() | First |
| `/ask` | API | POST → QueryPipeline.ask() | First |
| `/chat` | API | POST → QueryPipeline.ask_stream() (SSE) | First |
| `/*` | Static | Serve _book/ HTML files | Last (catch-all) |

Chat widget calls `/chat` (SSE streaming) on the same origin. No CORS needed. No API key in browser.

---

## Quarto & Widget Contract

| File | Purpose |
|---|---|
| `widget/_quarto-chat.yml` | Quarto profile overlay — adds `include-after-body: [_includes/chat.html]` |
| `widget/chat.html` | Chat UI — streams answers via `/chat` SSE, renders markdown, shows citations with book page links |

### Render flow

1. `quarto-builder` clones (first run) or pulls (subsequent) the book repo into `book_source` volume
2. Profile overlay injects chat widget into every page
3. `quarto render --profile chat` produces static HTML in `_book/`
4. Output persisted in `book_html` volume, mounted read-only in `app`
5. `app` serves `_book/` on the catch-all route

Trigger: `docker compose --profile build run quarto-builder`

The `indexer` reads from `book_source` but does not clone or pull — depends on `quarto-builder` having run first.

---

## Chunk Types

| Type | Source | Key metadata |
|---|---|---|
| Concept | Prose under a heading | chapter, section, heading_path, url |
| Code cell | Notebook code + explanation | chapter, section, cell_index, url |
| Figure | Caption + surrounding text | chapter, section, figure_id |
| Reference | BibTeX entry + context | citation_key |
| Paper | PDF content | chapter, source_path |
| Video segment | Transcript with timestamp | video_id, timestamp_seconds, watch_link |
| Chapter summary | Generated or authored summary | chapter |

Every chunk carries: `source_type`, `content_type`, `content_hash`, `commit_sha`.

### Parent/Child Relationships

Child chunks (~500w) optimize retrieval precision. Parent chunks (~2500w) optimize LLM context richness. Linked via `parent_id` and `chunk_type` metadata.

```mermaid
graph TD
  subgraph "e.g. Book Chapter Section"
    Parent["Parent chunk (~2500w)<br>chunk_type=parent<br>id=abc123"]
    Child1["Child chunk (~500w)<br>chunk_type=child<br>parent_id=abc123"]
    Child2["Child chunk (~500w)<br>chunk_type=child<br>parent_id=abc123<br>← retrieved by search"]
    Child3["Child chunk (~500w)<br>chunk_type=child<br>parent_id=abc123"]
  end

  Child2 -->|"parent expansion:<br>look up parent_id"| Parent
  Parent -->|"returned to LLM<br>(richer context)"| LLM[ContextBuilder]

  style Child2 fill:#fef3c7
  style Parent fill:#d1fae5
```

| Content Type | Parent | Child |
|---|---|---|
| Book chapter (.md/.qmd) | Full section (h2 block, ~2500w) | Paragraphs within (~500w) |
| Notebook (.ipynb) | Cell group (markdown + code) | Individual cells |
| Video transcript | ~3-5 min segment | ~30-60s segment (~180w) |
| Bibliography (.bib) | N/A (already small) | One entry per chunk |

---

## Model Strategy

All models swappable via env vars. Validated via eval harness.

### Embedding Models

| Model | Size | Strengths | Use when |
|---|---|---|---|
| **BAAI/bge-large-en-v1.5** | 335M | Battle-tested, CPU-viable, good MTEB | **Default** |
| **BAAI/bge-m3** | 568M | Native dense + sparse | Simpler hybrid search |
| **Qwen3-Embedding-8B** | 8B | Highest open-source MTEB (70.6) | GPU batch indexing |

Swap via `EMBEDDING_MODEL_NAME` env var.

### LLM Models (Generation)

| Model | Size | Runtime | Use when |
|---|---|---|---|
| **NASA proxy** | — | API (`proxy.fast.luna.nasa.gov/v1`) | **Production default** |
| **Qwen3-8B** | 8B | Ollama (MLX on Mac Silicon) | **Local dev default** |
| **Llama 3.3-8B** | 8B | Ollama | Alternative local |
| **Qwen3-14B/32B** | 14-32B | Ollama | Mac with 32GB+ RAM |

All OpenAI-compatible — same adapter. Swap via `LLM_BASE_URL` + `LLM_MODEL`.

### Reranker Models

| Model | Size | Strengths | Use when |
|---|---|---|---|
| **NoOpReranker** | — | Pass-through | **Default** |
| **BAAI/bge-reranker-v2-m3** | 568M | Multilingual, CPU-viable, Apache 2.0 | **First upgrade** |
| **Qwen3-Reranker-4B** | 4B | Strongest open-source, 32k context | Quality over speed |

Swap via `RERANKER_PROVIDER` env var.

### Chunking

| Strategy | Description |
|---|---|
| **Recursive (SectionChunker)** | Split by headings → paragraphs → sentences. Best overall. |
| **Parent/child** | 500w children for retrieval, 2500w parents for context. |

### Transcription

| Model | Library | Notes |
|---|---|---|
| **Whisper large-v3** | openai-whisper | Pre-compute on CPU, commit to repo |
| **distil-whisper-large-v3** | transformers | 6x faster, slight quality trade-off |

### Key Libraries

| Library | Covers | Why |
|---|---|---|
| **sentence-transformers** | Embeddings + reranking | 15k+ models, unified API |
| **Ollama** | Local LLM serving | OpenAI-compatible API, MLX on Apple Silicon |
| **qdrant-client** | Vector store | Native hybrid search, first-class Python SDK |
| **openai-whisper** | Transcription | Best ecosystem |
