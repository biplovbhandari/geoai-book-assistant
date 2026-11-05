from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from geoai_rag.models.chunk import Chunk
from geoai_rag.models.citation import Citation
from geoai_rag.models.generation_result import GenerationResult, StreamingGenerationResult
from geoai_rag.models.retrieval_result import RetrievalResult
from geoai_rag.models.scored_chunk import ScoredChunk


@runtime_checkable
class Embedder(Protocol):
    """Converts text into dense vector embeddings."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: The query text.

        Returns:
            Embedding vector.
        """
        ...

    def get_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...


@runtime_checkable
class SparseEmbedder(Protocol):
    """Converts text into sparse vector embeddings."""

    @property
    def requires_idf(self) -> bool:
        """Whether the model needs Qdrant's IDF modifier on the sparse vector config.

        BM25 and BM42 produce raw term frequencies and need server-side IDF
        weighting. SPLADE produces learned weights and does not.
        """
        ...

    def embed(self, texts: list[str]) -> list[Any]:
        """Embed a batch of document texts for indexing.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of sparse embedding objects (one per input text),
            each with .indices and .values attributes.
        """
        ...

    def query_embed(self, text: str) -> list[Any]:
        """Embed a single query string for search.

        Args:
            text: The query text.

        Returns:
            List containing a single sparse embedding object.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Persistent store for chunk vectors and payloads."""

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or update chunks with their embedding vectors.

        Args:
            chunks: Chunk objects (payload stored alongside vectors).
            vectors: Corresponding embedding vectors (same length as chunks).
        """
        ...

    def search_dense(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Search by dense vector similarity.

        Args:
            vector: Query embedding vector.
            top_k: Maximum number of results.
            filters: Optional metadata filters.

        Returns:
            Ranked list of scored chunks.
        """
        ...

    def search_sparse(
        self,
        text: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Search by sparse/keyword matching (e.g. BM25).

        Args:
            text: Raw query text.
            top_k: Maximum number of results.
            filters: Optional metadata filters.

        Returns:
            Ranked list of scored chunks.
        """
        ...

    def get_by_ids(self, ids: list[str]) -> list[Chunk]:
        """Retrieve chunks by their IDs.

        Args:
            ids: List of chunk IDs.

        Returns:
            List of matching chunks.
        """
        ...

    def delete_by_source(self, source_path: str) -> None:
        """Delete all chunks originating from a given source file.

        Args:
            source_path: Repo-relative path (e.g. "book/03_Semantic_Segmentation/...").
        """
        ...

    def count(self) -> int:
        """Return the approximate number of points in the collection."""
        ...


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Retrieves relevant chunks for a query."""

    def retrieve(
        self,
        question: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Retrieve ranked chunks for a natural language question.

        Args:
            question: The user's query.
            top_k: Maximum number of results.
            filters: Optional metadata filters.

        Returns:
            Ranked chunks together with the per-stage trace of how they were produced.
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Re-scores and re-orders retrieval candidates."""

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        """Rerank candidates and return the top results.

        Args:
            query: The original query text.
            candidates: Chunks from the initial retrieval.
            top_k: Maximum number of results to return.

        Returns:
            Reranked list of scored chunks.
        """
        ...


@runtime_checkable
class LLMClient(Protocol):
    """Generates text from a chat message list."""

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """Generate a response from a chat message list.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the generated response.

        Returns:
            The generated text plus token counts, finish reason, and model name.
        """
        ...

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> StreamingGenerationResult:
        """Stream a response as text chunks, with metadata available after iteration.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the generated response.

        Returns:
            An iterable of text chunks as the model produces them. Once fully
            iterated, its ``metadata`` attribute holds token counts, finish
            reason, and model name.
        """
        ...


@runtime_checkable
class ContextBuilder(Protocol):
    """Assembles retrieval chunks into an LLM message list."""

    def build(
        self,
        question: str,
        chunks: list[ScoredChunk],
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Assemble retrieved chunks into a message list for the LLM.

        Args:
            question: The user's natural language question.
            chunks: Ranked chunks from retrieval.
            history: Optional prior conversation turns for reference resolution.

        Returns:
            List of message dicts (system + user) ready for LLMClient.chat().
        """
        ...


@runtime_checkable
class CitationBuilder(Protocol):
    """Extracts citation metadata from retrieval chunks."""

    def build(self, chunks: list[ScoredChunk]) -> list[Citation]:
        """Extract citation metadata from retrieved chunks.

        Args:
            chunks: Ranked chunks from retrieval.

        Returns:
            List of citations, one per chunk.
        """
        ...
