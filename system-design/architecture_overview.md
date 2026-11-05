# Geo-AI Book RAG Assistant - Architecture Overview

**Book:** [nasa-earthrise.github.io/EarthRISE-Applied-Artificial-Intelligence-and-Deep-Learning-Book](https://nasa-earthrise.github.io/EarthRISE-Applied-Artificial-Intelligence-and-Deep-Learning-Book/) \
**Videos:** [YouTube Playlist](https://www.youtube.com/playlist?list=PLKlxghiZuIM5YyM92lDtsJT7p0Aln3R_k)

---

## Design Principles

| Principle | Meaning |
|---|---|
| Interface-driven | Key components sit behind interfaces. Swap adapters via env vars — zero code changes. |
| Retrieval quality first | Bad retrieval + great LLM = hallucinations. Great retrieval + decent LLM = good answers. |
| Components are independent | Embedding, storage, generation, and retrieval are separate adapters. Mix NASA infra with our own per-component. |

---

## System Overview

```mermaid
graph TD
  User[User] --> App

  subgraph "Online Services"
    App["FastAPI App<br/>Quarto HTML + /ask /search /chat /health"]
    DB[(Qdrant<br/>vector + keyword search)]
    EmbedQ[Embedder<br/>query-time]
  end

  subgraph "Offline Services"
    Quarto["quarto-builder<br/>git pull → render → _book/"]
    Index["indexer<br/>parse → chunk → embed → store"]
  end

  subgraph "NASA Services"
    ChatGSFC["NASA Proxy<br/>LLM generation"]
  end

  App --> EmbedQ --> DB
  App --> ChatGSFC
  Quarto -->|"_book/"| App
  Index --> DB

  style App fill:#1a5276,color:#fff
  style ChatGSFC fill:#fadbd8
  style DB fill:#fdebd0
  style Index fill:#d5f5e3
  style Quarto fill:#d5f5e3
  style EmbedQ fill:#d6eaf8
```

---

## How a Question Gets Answered

```mermaid
sequenceDiagram
  participant User
  participant App as FastAPI App
  participant DB as Qdrant
  participant LLM as LLM (API)

  User->>App: Chat widget POST /chat {question, history}
  App->>App: Embed query
  App->>DB: Dense vector search
  App->>DB: Keyword / BM25 search
  DB-->>App: Candidate chunks
  App->>App: Merge + rerank → top chunks
  App->>App: Build context from chunks
  App->>LLM: Generate answer (API call)
  LLM-->>App: Answer
  App->>App: Add citations + video links
  App-->>User: SSE stream: meta → tokens → done
```

---

## How Content Gets Indexed

```mermaid
graph TD
  subgraph Sources
    Book[Book repo<br>.md .qmd .ipynb .bib]
    Videos[YouTube lectures]
    Other[Figures, PDFs]
  end

  subgraph "indexer (offline)"
    Parse[Parse sources]
    Chunk[Chunk into<br>parent/child pairs]
    Embed[Embed]
  end

  Store[(Qdrant)]

  Book --> Parse
  Videos --> Parse
  Other -.-> Parse
  Parse --> Chunk --> Embed --> Store

  style Other fill:#f5f5f5,stroke-dasharray: 5 5
  style Videos fill:#f5f5f5,stroke-dasharray: 5 5
  style Store fill:#fdebd0
```

---

## Infrastructure

```mermaid
graph TD
  subgraph "Online (always running)"
    app["app — FastAPI<br/>serves book HTML + RAG API"]
    qdrant["qdrant — Vector DB"]
  end
  subgraph "Offline (on-demand)"
    qb["quarto-builder<br/>renders book with chat widget"]
    idx["indexer<br/>parse → chunk → embed → store"]
  end
  qb -->|"_book/"| app
  idx --> qdrant
```

| Container | Role | Online? |
|---|---|---|
| **app** | FastAPI: serves _book/ (Quarto HTML) + RAG API. Same origin. | Yes |
| **qdrant** | Vector DB (dense + sparse search) | Yes |
| **quarto-builder** | git pull → inject chat widget → quarto render → _book/ | No — offline |
| **indexer** | parse → chunk → embed → upsert to Qdrant | No — offline |

---

## What's Swappable

```mermaid
graph LR
  subgraph Generation
    A1["NASA Proxy"] ~~~ A2["Ollama (local)"]
  end
  subgraph Embedding
    B1["bge-large-en-v1.5"] ~~~ B2["Qwen3-Embedding-8B"]
  end
  subgraph Storage
    C1[Qdrant]
  end
  subgraph Retrieval
    D0[Dense<br>vector search] ~~~ D1[Hybrid<br>dense + BM25]
    D1 -->|"wrapped by"| D2[Parent/Child<br>decorator]
  end

  style A1 fill:#d5f5e3
  style B1 fill:#d5f5e3
  style C1 fill:#d5f5e3
  style D0 fill:#d5f5e3
  style A2 fill:#f5f5f5
  style B2 fill:#f5f5f5
  style D2 fill:#d6eaf8
```

Green = current default. Gray = alternatives. Swap via env vars.

---

## How the Book Gets Served

```mermaid
graph LR
  Repo["Book repo<br/>(GitHub)"] -->|"git pull"| QB["quarto-builder"]
  Widget["_quarto-chat.yml<br/>+ chat.html"] -->|"profile overlay"| QB
  QB -->|"quarto render"| Book["_book/<br/>(static HTML)"]
  Book -->|"mounted"| App["FastAPI app<br/>catch-all route"]
  App -->|"serves"| User["Browser"]
```

Chat widget calls `/chat` (SSE streaming) on the same origin — no CORS, no API key in browser.

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service status |
| `/search` | POST | Retrieval-only — ranked chunks with metadata |
| `/ask` | POST | Full RAG — generated answer + citations |
| `/chat` | POST | Streaming RAG — SSE token-by-token answer + citations |

Additional endpoints:
- `/feedback` — user rating on answer quality
- `/log` — analytics data forwarding
- `/metrics` — Prometheus metrics (admin-gated)
- `/debug/*` - pipeline trace, retrieval inspector, and groundedness (admin-gated)

---

## Pipeline Factories and Readiness

`geoai_rag.factories` provides pipeline factory functions (`create_pipelines()`, `create_indexing_pipeline()`, and per-component factories).
The API layer and CLI scripts both import from this module, keeping pipeline construction in one place.

Routes declare readiness requirements via FastAPI `dependencies=[...]`.
`require_retrieval_ready` and `require_generation_ready` (in `api.dependencies`) return 503 when the required adapters are not configured.
`QueryPipeline.generation_ready` is the single source of truth for whether all generation adapters are available.

---

## Analytics & Economic Impact Assessment (EIA)

GA4 custom events and Google Sheets integration for the Economic Impact Assessment.

| Event | Trigger | Purpose |
|---|---|---|
| `question_asked` | Question sent via chat widget | Usage tracking |
| `chapter_complete` | Scroll > 90% | Engagement |
| `colab_click` | Notebook link clicked | Active learning signal |
| `video_deeplink_click` | Timestamp link clicked | Video engagement |
| `survey_role` | Role button tapped | User segmentation |
| `survey_usefulness` | Rating tapped | Rubric scoring |

---

## Security

- API keys server-side only (never in browser)
- Same-origin serving eliminates client-side API key exposure
- System prompt treats retrieved content as reference material, not instructions
- `LLM_API_KEY` stored as `SecretStr` — never appears in logs

---

## Observability

**Structured logging.**
JSON output in production (`python-json-logger`), plain text in development.
Every log line includes a request ID propagated via `contextvars`, matching the `X-Request-ID` response header for end-to-end correlation.

**Monitoring stack.**
Prometheus scrapes both the Geo-AI Book Assistant app (`/metrics`) and Qdrant's built-in `/metrics` endpoint.
Grafana reads from Prometheus and loads an auto-provisioned dashboard.
Both services bind to `127.0.0.1` only - not exposed externally.

---

For detailed component contracts, interfaces, and adapter specifications see [architecture_contracts.md](architecture_contracts.md).
