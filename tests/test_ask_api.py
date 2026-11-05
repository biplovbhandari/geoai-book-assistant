from conftest import (
    create_test_client,
    make_retrieval_result,
    make_scored_chunk,
    make_streaming_result,
)

from geoai_rag.models.answer import Answer
from geoai_rag.models.citation import Citation
from geoai_rag.models.generation_result import GenerationResult
from geoai_rag.models.pipeline_result import PipelineResult


def _make_answer(chunks=None):
    if chunks is None:
        chunks = [make_scored_chunk()]
    return Answer(
        answer="U-Net is a convolutional neural network [1].",
        sources=chunks,
        citations=[
            Citation(
                chunk_id=sc.chunk.id,
                source_path=sc.chunk.metadata.get("source_path", ""),
                chapter=sc.chunk.metadata.get("chapter", ""),
                section=sc.chunk.metadata.get("section", ""),
                display_label="03 - U-Net",
            )
            for sc in chunks
        ],
    )


def _make_pipeline_result(chunks=None):
    """Wrap a canned Answer in the PipelineResult that QueryPipeline.ask() returns."""
    answer = _make_answer(chunks)
    return PipelineResult(
        answer=answer,
        retrieval_trace=make_retrieval_result(answer.sources).trace,
        generation=GenerationResult(text=answer.answer),
    )


class _FakeQueryPipeline:
    generation_ready = True

    def __init__(self, ask_result):
        self._context_builder = object()
        self._llm_client = self
        self._citation_builder = object()
        self._ask_result = ask_result

    def ask(self, question, filters=None, *, history=None):
        return self._ask_result

    def chat_stream(self, messages, temperature=0.3, max_tokens=1024):
        return make_streaming_result(["test"])


class _FakeFailingQueryPipeline(_FakeQueryPipeline):
    def ask(self, question, filters=None, *, history=None):
        raise RuntimeError("LLM timeout")


def _make_fake_pipelines(ask_result=None, count=100):
    from api.dependencies import Pipelines

    query = _FakeQueryPipeline(ask_result or _make_pipeline_result())
    store = type("FakeStore", (), {"count": lambda self: count})()

    return Pipelines(query=query, vector_store=store)  # type: ignore[arg-type]


class TestAsk:
    def test_valid_query_returns_answer(self, monkeypatch):
        client = create_test_client(monkeypatch, _make_fake_pipelines())
        with client:
            resp = client.post("/ask", json={"question": "What is U-Net?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "sources" in body
        assert "citations" in body
        assert len(body["sources"]) == 1
        assert len(body["citations"]) == 1
        assert body["citations"][0]["display_label"]

    def test_llm_failure_returns_503(self, monkeypatch):
        from api.dependencies import Pipelines

        query = _FakeFailingQueryPipeline(_make_pipeline_result())
        store = type("FakeStore", (), {"count": lambda self: 100})()
        pipelines = Pipelines(query=query, vector_store=store)  # type: ignore[arg-type]
        client = create_test_client(monkeypatch, pipelines)
        with client:
            resp = client.post("/ask", json={"question": "test"})
        assert resp.status_code == 503
