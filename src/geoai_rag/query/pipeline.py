from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from geoai_rag.interfaces import (
    CitationBuilder,
    ContextBuilder,
    LLMClient,
    RetrievalStrategy,
)
from geoai_rag.models.answer import Answer
from geoai_rag.models.scored_chunk import ScoredChunk

_MAX_AUGMENT_CHARS = 500


class QueryPipeline:
    """Composition root for query operations (search and ask)."""

    def __init__(
        self,
        strategy: RetrievalStrategy,
        context_builder: ContextBuilder | None = None,
        llm_client: LLMClient | None = None,
        citation_builder: CitationBuilder | None = None,
        top_k: int = 8,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize the query pipeline.

        Args:
            strategy: Retrieval strategy for finding relevant chunks.
            context_builder: Assembles chunks into LLM messages (None disables ask).
            llm_client: Generates answers from messages (None disables ask).
            citation_builder: Extracts citation metadata from chunks (None disables ask).
            top_k: Number of chunks to retrieve for ask() (from settings).
            max_tokens: Maximum tokens for generated answers (from settings).
        """
        self._strategy = strategy
        self._context_builder = context_builder
        self._llm_client = llm_client
        self._citation_builder = citation_builder
        self._top_k = top_k
        self._max_tokens = max_tokens

    @property
    def generation_ready(self) -> bool:
        """Whether all generation adapters are configured."""
        return (
            self._context_builder is not None
            and self._llm_client is not None
            and self._citation_builder is not None
        )

    def search(
        self,
        question: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Search for relevant chunks using the configured retrieval strategy.

        Args:
            question: Natural language query.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.

        Returns:
            Ranked list of scored chunks.
        """
        return self._strategy.retrieve(question, top_k, filters)

    @staticmethod
    def _build_retrieval_query(
        question: str,
        history: list[dict[str, str]] | None,
    ) -> str:
        """Augment query with context from the last user message in history.

        Current question comes first so it survives embedding model truncation.
        Uses defensive dict access for direct pipeline callers that may pass
        malformed history dicts.

        Args:
            question: The current user question.
            history: Optional conversation history as a list of role/content dicts.

        Returns:
            The question, optionally appended with the most recent prior user message.
        """
        if not history:
            return question
        for msg in reversed(history):
            if msg.get("role") == "user":
                prev = str(msg.get("content", ""))[:_MAX_AUGMENT_CHARS]
                return f"{question} {prev}"
        return question

    def ask(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        *,
        history: list[dict[str, str]] | None = None,
    ) -> Answer:
        """Retrieve chunks, build context, call LLM, and return a cited answer.

        Args:
            question: The user's natural language question.
            filters: Optional metadata filters for retrieval.
            history: Optional conversation history as a list of role/content dicts.

        Returns:
            Answer with generated text, source chunks, citations, and the
            elapsed retrieval and generation times in milliseconds.

        Raises:
            RuntimeError: If generation adapters are not configured.
        """
        if not self.generation_ready:
            raise RuntimeError(
                "Generation not configured"
                " - context_builder, llm_client, and citation_builder are required"
            )
        assert self._context_builder is not None
        assert self._llm_client is not None
        assert self._citation_builder is not None

        retrieval_query = self._build_retrieval_query(question, history)
        t0 = time.monotonic()
        chunks = self._strategy.retrieve(retrieval_query, self._top_k, filters)
        retrieval_ms = int((time.monotonic() - t0) * 1000)

        if not chunks:
            return Answer(
                answer="No relevant information found in the book for this question.",
                sources=[],
                citations=[],
                retrieval_ms=retrieval_ms,
                generation_ms=0,
            )

        messages = self._context_builder.build(question, chunks, history=history)
        t1 = time.monotonic()
        answer_text = self._llm_client.chat(messages, max_tokens=self._max_tokens)
        generation_ms = int((time.monotonic() - t1) * 1000)
        citations = self._citation_builder.build(chunks)

        return Answer(
            answer=answer_text,
            sources=chunks,
            citations=citations,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    def ask_stream(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        filters: dict[str, Any] | None = None,
        _recording_ctx: dict | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream a RAG answer as SSE-compatible events.

        Yields dicts with a ``type`` key: ``meta`` (citations), ``token``
        (text chunk), ``error``, or ``done``.  The ``meta`` event is always
        emitted first, even for zero-chunk responses.

        Readiness is NOT checked here -- the route handler must verify all
        adapters before constructing ``StreamingResponse``.

        Args:
            question: The user's natural language question.
            history: Optional conversation history as a list of role/content dicts.
            filters: Optional metadata filters for retrieval.
            _recording_ctx: Optional mutable dict populated with retrieval data
                and timing for database recording. Keys set: scored_chunks,
                retrieval_query, retrieval_ms, generation_wall_ms. Does not
                affect the yielded event stream.

        Yields:
            Event dicts with a ``type`` key and type-specific fields.

        Raises:
            RuntimeError: If generation adapters are not configured.
        """
        if not self.generation_ready:
            raise RuntimeError(
                "Generation not configured"
                " - context_builder, llm_client, and citation_builder are required"
            )
        assert self._context_builder is not None
        assert self._llm_client is not None
        assert self._citation_builder is not None

        retrieval_query = self._build_retrieval_query(question, history)
        t0 = time.monotonic()
        chunks = self._strategy.retrieve(retrieval_query, self._top_k, filters)
        retrieval_ms = int((time.monotonic() - t0) * 1000)

        if not chunks:
            if _recording_ctx is not None:
                _recording_ctx["scored_chunks"] = []
                _recording_ctx["retrieval_query"] = retrieval_query
                _recording_ctx["retrieval_ms"] = retrieval_ms
                _recording_ctx["generation_wall_ms"] = 0
            yield {"type": "meta", "citations": []}
            yield {
                "type": "token",
                "content": "No relevant information found in the book for this question.",
            }
            yield {"type": "done"}
            return

        if _recording_ctx is not None:
            _recording_ctx["scored_chunks"] = [
                {
                    "chunk_id": sc.chunk.id,
                    "content": sc.chunk.content,
                    "source_type": sc.chunk.source_type,
                    "metadata": sc.chunk.metadata,
                    "score": sc.score,
                    "ranking_method": sc.ranking_method,
                }
                for sc in chunks
            ]
            _recording_ctx["retrieval_query"] = retrieval_query
            _recording_ctx["retrieval_ms"] = retrieval_ms

        citations = self._citation_builder.build(chunks)
        yield {"type": "meta", "citations": [c.model_dump() for c in citations]}

        messages = self._context_builder.build(question, chunks, history=history)

        has_content = False
        t1 = time.monotonic()
        for token in self._llm_client.chat_stream(messages, max_tokens=self._max_tokens):
            yield {"type": "token", "content": token}
            if token.strip():
                has_content = True
        generation_wall_ms = int((time.monotonic() - t1) * 1000)

        if _recording_ctx is not None:
            _recording_ctx["generation_wall_ms"] = generation_wall_ms

        if not has_content:
            yield {"type": "error", "message": "The model returned an empty response."}
            return

        yield {"type": "done"}
