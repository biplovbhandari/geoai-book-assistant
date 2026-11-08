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
from geoai_rag.models.generation_result import GenerationResult
from geoai_rag.models.pipeline_result import PipelineResult
from geoai_rag.models.retrieval_result import RetrievalResult

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
    ) -> RetrievalResult:
        """Search for relevant chunks using the configured retrieval strategy.

        Args:
            question: Natural language query.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.

        Returns:
            Ranked chunks together with the per-stage retrieval trace.
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
    ) -> PipelineResult:
        """Retrieve chunks, build context, call the LLM, and return the answer with its trace.

        Args:
            question: The user's natural language question.
            filters: Optional metadata filters for retrieval.
            history: Optional conversation history as a list of role/content dicts.

        Returns:
            PipelineResult holding the widget-facing ``Answer`` (generated text,
            source chunks, citations, and the elapsed retrieval and generation
            times in milliseconds), the retrieval trace, the generation
            metadata, and the messages sent to the LLM. When retrieval finds no
            chunks, the generation metadata is empty and no messages are
            recorded because the LLM is never called.

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
        result = self._strategy.retrieve(retrieval_query, self._top_k, filters)
        retrieval_ms = int((time.monotonic() - t0) * 1000)

        # Check result.chunks, not result: a RetrievalResult is always truthy.
        if not result.chunks:
            empty_answer = Answer(
                answer="No relevant information found in the book for this question.",
                sources=[],
                citations=[],
                retrieval_ms=retrieval_ms,
                generation_ms=0,
            )
            return PipelineResult(
                answer=empty_answer,
                retrieval_trace=result.trace,
                generation=GenerationResult(text=""),
                context_messages=[],
            )

        messages = self._context_builder.build(question, result.chunks, history=history)
        t1 = time.monotonic()
        gen_result = self._llm_client.chat(messages, max_tokens=self._max_tokens)
        generation_ms = int((time.monotonic() - t1) * 1000)
        citations = self._citation_builder.build(result.chunks)

        answer = Answer(
            answer=gen_result.text,
            sources=result.chunks,
            citations=citations,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )
        return PipelineResult(
            answer=answer,
            retrieval_trace=result.trace,
            generation=gen_result,
            context_messages=messages,
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
            _recording_ctx: Optional mutable dict populated with retrieval data,
                trace data, and timing for database recording. Does not affect
                the yielded event stream. Keys set:

                - ``scored_chunks``, ``retrieval_query``, ``retrieval_ms``,
                  ``retrieval_trace`` (dict): after retrieval, on every path.
                - ``context_messages``: the messages sent to the LLM, set as
                  soon as they are built. Unset when no chunks were found.
                - ``generation_wall_ms``: after the LLM stream ends. Set to 0
                  when no chunks were found.
                - ``generation_metadata`` (dict): after the LLM stream ends,
                  when the stream reports it. Unset when no chunks were found.

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
        result = self._strategy.retrieve(retrieval_query, self._top_k, filters)
        retrieval_ms = int((time.monotonic() - t0) * 1000)

        # Check result.chunks, not result: a RetrievalResult is always truthy.
        if not result.chunks:
            if _recording_ctx is not None:
                _recording_ctx["scored_chunks"] = []
                _recording_ctx["retrieval_query"] = retrieval_query
                _recording_ctx["retrieval_ms"] = retrieval_ms
                _recording_ctx["retrieval_trace"] = result.trace.model_dump()
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
                for sc in result.chunks
            ]
            _recording_ctx["retrieval_query"] = retrieval_query
            _recording_ctx["retrieval_ms"] = retrieval_ms
            _recording_ctx["retrieval_trace"] = result.trace.model_dump()

        citations = self._citation_builder.build(result.chunks)
        yield {"type": "meta", "citations": [c.model_dump() for c in citations]}

        messages = self._context_builder.build(question, result.chunks, history=history)
        if _recording_ctx is not None:
            _recording_ctx["context_messages"] = messages

        has_content = False
        # Start the clock before chat_stream() so the timing still covers the
        # request setup, which happens at call time rather than during iteration.
        t1 = time.monotonic()
        stream = self._llm_client.chat_stream(messages, max_tokens=self._max_tokens)
        for token in stream:
            yield {"type": "token", "content": token}
            if token.strip():
                has_content = True
        generation_wall_ms = int((time.monotonic() - t1) * 1000)

        # stream.metadata is only populated once the stream is fully consumed.
        if _recording_ctx is not None:
            _recording_ctx["generation_wall_ms"] = generation_wall_ms
            if stream.metadata is not None:
                _recording_ctx["generation_metadata"] = stream.metadata.model_dump()

        if not has_content:
            yield {"type": "error", "message": "The model returned an empty response."}
            return

        done_event: dict = {"type": "done"}
        if stream.metadata is not None:
            if stream.metadata.prompt_tokens is not None:
                done_event["prompt_tokens"] = stream.metadata.prompt_tokens
            if stream.metadata.completion_tokens is not None:
                done_event["completion_tokens"] = stream.metadata.completion_tokens
        yield done_event
