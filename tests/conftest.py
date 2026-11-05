"""Shared test fixtures, factories, and fakes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from geoai_rag.models import Chunk, ScoredChunk
from geoai_rag.models.citation import Citation
from geoai_rag.models.generation_result import StreamingGenerationResult
from geoai_rag.models.retrieval_result import RetrievalResult, RetrievalTrace


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear rate limiter state between tests to prevent cross-test 429s."""
    from api.middleware import RateLimitMiddleware

    try:
        from api.main import app
    except ImportError:
        return
    current = getattr(app, "middleware_stack", None)
    while current is not None:
        if isinstance(current, RateLimitMiddleware):
            current._requests.clear()
            return
        current = getattr(current, "app", None)


def make_scored_chunk(
    content="U-Net architecture",
    score=0.95,
    method="dense",
    metadata=None,
):
    """Create a ScoredChunk for testing."""
    chunk = Chunk(
        content=content,
        content_hash="abc",
        source_type="book_text",
        content_type="concept",
        metadata=metadata
        or {
            "source_path": "book/03_Segmentation/index.qmd",
            "chapter": "03",
            "section": "U-Net",
        },
    )
    return ScoredChunk(chunk=chunk, score=score, ranking_method=method)


def make_retrieval_result(chunks, query="test"):
    """Wrap chunks in a minimal RetrievalResult with an empty trace."""
    return RetrievalResult(
        chunks=chunks,
        trace=RetrievalTrace(stages=[], retrieval_query=query, total_latency_ms=0),
    )


def make_streaming_result(tokens):
    """Build a StreamingGenerationResult that yields the given text tokens.

    Appends a final chunk carrying usage and finish_reason, like a real
    OpenAI-compatible stream.
    """

    class _Chunk:
        def __init__(self, content, finish_reason=None, usage=None):
            delta = type("Delta", (), {"content": content})()
            choice = type("Choice", (), {"delta": delta, "finish_reason": finish_reason})()
            self.choices = [choice]
            self.usage = usage
            self.model = "test-model"

    usage = type("Usage", (), {"prompt_tokens": 100, "completion_tokens": 50})()
    chunks = [_Chunk(t) for t in tokens]
    chunks.append(_Chunk(None, finish_reason="stop", usage=usage))
    return StreamingGenerationResult(iter(chunks))


class FakeStrategy:
    """Canned retrieval strategy for testing."""

    def __init__(self, results):
        self._results = results

    def retrieve(self, question, top_k, filters=None):
        """Return canned results up to top_k, wrapped in a RetrievalResult."""
        return make_retrieval_result(self._results[:top_k], query=question)


class FakeContextBuilder:
    """Canned context builder for testing."""

    def build(self, question, chunks, history=None):
        """Return a fixed two-message context."""
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"Q: {question}"},
        ]


class FakeCitationBuilder:
    """Canned citation builder for testing."""

    def build(self, chunks):
        """Return one citation per chunk with a fixed URL."""
        citations = []
        for sc in chunks:
            chapter = sc.chunk.metadata.get("chapter", "")
            section = sc.chunk.metadata.get("section", "")
            if chapter and section:
                display_label = f"{chapter} - {section}"
            else:
                display_label = chapter or section or "Source"
            citations.append(
                Citation(
                    chunk_id=sc.chunk.id,
                    source_path=sc.chunk.metadata.get("source_path", ""),
                    chapter=chapter,
                    section=section,
                    url="/ch1.html",
                    display_label=display_label,
                )
            )
        return citations


def create_test_client(monkeypatch, pipelines=None):
    """Create a TestClient with mocked pipelines and DB disabled."""
    monkeypatch.setenv("DATABASE_URL", "")

    from geoai_rag.config import get_settings

    get_settings.cache_clear()

    from api.main import app

    if pipelines is not None:
        monkeypatch.setattr("api.main.create_pipelines", lambda config: pipelines)

    return TestClient(app)
