import json
import uuid

from conftest import create_test_client, make_streaming_result

from api.routes.chat import MAX_HISTORY_CONTENT
from geoai_rag.models.generation_result import GenerationResult


class FakeStreamingQueryPipeline:
    """Explicit fake with all required adapters for readiness checks."""

    generation_ready = True

    def __init__(self):
        self._context_builder = object()
        self._llm_client = self  # self has chat_stream
        self._citation_builder = object()
        self.last_history = None

    def ask_stream(self, question, *, history=None, filters=None, _recording_ctx=None):
        self.last_history = history
        yield {"type": "meta", "citations": []}
        yield {"type": "token", "content": "Answer text."}
        yield {"type": "done"}

    def chat_stream(self, messages, temperature=0.3, max_tokens=1024):
        return make_streaming_result(["test"])


class FakeNonStreamingQueryPipeline:
    """Fake WITHOUT chat_stream -- must produce 503."""

    generation_ready = True

    def __init__(self):
        self._context_builder = object()
        self._llm_client = _NoStreamLLM()
        self._citation_builder = object()


class _NoStreamLLM:
    def chat(self, messages, temperature=0.3, max_tokens=1024):
        return GenerationResult(text="answer")


class FakeNonCallableStreamQueryPipeline:
    """Fake with chat_stream as a non-callable attribute -- must produce 503."""

    generation_ready = True

    def __init__(self):
        self._context_builder = object()
        self._llm_client = _NonCallableStreamLLM()
        self._citation_builder = object()


class _NonCallableStreamLLM:
    chat_stream = "not callable"

    def chat(self, messages, temperature=0.3, max_tokens=1024):
        return GenerationResult(text="answer")


class FakeIncompleteQueryPipeline:
    """Fake with _context_builder=None -- must produce 503."""

    generation_ready = False

    def __init__(self):
        self._context_builder = None
        self._llm_client = object()
        self._citation_builder = object()


class FakeVectorStore:
    def __init__(self, count_val=100):
        self._count = count_val

    def count(self):
        return self._count


def _make_fake_pipelines(query=None, store=None):
    from api.dependencies import Pipelines

    return Pipelines(
        query=query or FakeStreamingQueryPipeline(),  # type: ignore[arg-type]
        vector_store=store or FakeVectorStore(),  # type: ignore[arg-type]
    )


class TestChatEndpoint:
    def test_streams_sse_events(self, monkeypatch):
        client = create_test_client(monkeypatch, _make_fake_pipelines())
        with client:
            resp = client.post("/chat", json={"question": "What is U-Net?"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line.removeprefix("data: "))
            for line in resp.text.strip().split("\n\n")
            if line.startswith("data:")
        ]
        types = [e["type"] for e in events]
        assert "meta" in types
        assert "token" in types
        assert "done" in types

    def test_no_streaming_returns_503(self, monkeypatch):
        pipelines = _make_fake_pipelines(query=FakeNonStreamingQueryPipeline())
        client = create_test_client(monkeypatch, pipelines)
        with client:
            resp = client.post("/chat", json={"question": "Q?"})
        assert resp.status_code == 503
        assert resp.json()["detail"] == "streaming not supported"

    def test_incomplete_adapters_returns_503(self, monkeypatch):
        pipelines = _make_fake_pipelines(query=FakeIncompleteQueryPipeline())
        client = create_test_client(monkeypatch, pipelines)
        with client:
            resp = client.post("/chat", json={"question": "Q?"})
        assert resp.status_code == 503
        assert resp.json()["detail"] == "generation adapters incomplete"

    def test_empty_vector_store_returns_503(self, monkeypatch):
        pipelines = _make_fake_pipelines(store=FakeVectorStore(count_val=0))
        client = create_test_client(monkeypatch, pipelines)
        with client:
            resp = client.post("/chat", json={"question": "Q?"})
        assert resp.status_code == 503
        assert resp.json()["detail"] == "retrieval not ready"

    def test_history_truncation(self, monkeypatch):
        fake_query = FakeStreamingQueryPipeline()
        pipelines = _make_fake_pipelines(query=fake_query)
        client = create_test_client(monkeypatch, pipelines)
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(15)
        ]
        with client:
            resp = client.post("/chat", json={"question": "Q?", "history": history})
        assert resp.status_code == 200
        assert fake_query.last_history is not None
        assert len(fake_query.last_history) <= 10
        assert fake_query.last_history[0]["role"] == "user"

    def test_non_callable_chat_stream_returns_503(self, monkeypatch):
        pipelines = _make_fake_pipelines(query=FakeNonCallableStreamQueryPipeline())
        client = create_test_client(monkeypatch, pipelines)
        with client:
            resp = client.post("/chat", json={"question": "Q?"})
        assert resp.status_code == 503
        assert resp.json()["detail"] == "streaming not supported"

    def test_long_history_content_truncated_to_cap(self, monkeypatch):
        fake_query = FakeStreamingQueryPipeline()
        pipelines = _make_fake_pipelines(query=fake_query)
        client = create_test_client(monkeypatch, pipelines)
        history = [
            {"role": "user", "content": "short question"},
            {"role": "assistant", "content": "x" * 10000},
        ]
        with client:
            resp = client.post("/chat", json={"question": "Follow up?", "history": history})
        assert resp.status_code == 200
        # ChatMessage.content has no max_length, so a 200 alone is a tautology;
        # assert the 8000-char truncation actually reached the pipeline.
        assert fake_query.last_history is not None
        assert len(fake_query.last_history[1]["content"]) == MAX_HISTORY_CONTENT

    def test_whitespace_question_returns_422(self, monkeypatch):
        # "   " passes raw min_length=1 but our strip_question collapses it to "" -> 422,
        # so this exercises strip_question rather than Pydantic's min_length alone.
        client = create_test_client(monkeypatch, _make_fake_pipelines())
        with client:
            resp = client.post("/chat", json={"question": "   "})
        assert resp.status_code == 422

    def test_stream_exception_yields_error_event(self, monkeypatch):
        """A mid-stream pipeline exception is surfaced as an SSE error event, not a 500."""

        class _RaisingPipeline:
            generation_ready = True

            def __init__(self):
                self._context_builder = object()
                self._llm_client = self
                self._citation_builder = object()

            def ask_stream(self, question, *, history=None, filters=None, _recording_ctx=None):
                yield {"type": "meta", "citations": []}
                raise RuntimeError("boom")

            def chat_stream(self, messages, temperature=0.3, max_tokens=1024):
                return make_streaming_result(["x"])

        pipelines = _make_fake_pipelines(query=_RaisingPipeline())
        client = create_test_client(monkeypatch, pipelines)
        with client:
            resp = client.post("/chat", json={"question": "Q?"})
        assert resp.status_code == 200
        events = [
            json.loads(line.removeprefix("data: "))
            for line in resp.text.strip().split("\n\n")
            if line.startswith("data:")
        ]
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["message"] == "Generation failed. Please try again."

    def test_meta_includes_interaction_id(self, monkeypatch):
        """Meta event always includes interaction_id as a string UUID."""
        client = create_test_client(monkeypatch, _make_fake_pipelines())
        with client:
            resp = client.post("/chat", json={"question": "Q?"})
        events = [
            json.loads(line.removeprefix("data: "))
            for line in resp.text.strip().split("\n\n")
            if line.startswith("data:")
        ]
        meta = next(e for e in events if e["type"] == "meta")
        assert "interaction_id" in meta
        # Should be a valid UUID string
        uuid.UUID(meta["interaction_id"])

    def test_new_conversation_meta_includes_ids(self, monkeypatch):
        """First request (no conversation_id) gets conversation_id and visitor_id in meta."""
        client = create_test_client(monkeypatch, _make_fake_pipelines())
        with client:
            resp = client.post("/chat", json={"question": "Q?"})
        events = [
            json.loads(line.removeprefix("data: "))
            for line in resp.text.strip().split("\n\n")
            if line.startswith("data:")
        ]
        meta = next(e for e in events if e["type"] == "meta")
        assert "conversation_id" in meta
        assert "visitor_id" in meta
        uuid.UUID(meta["conversation_id"])
        uuid.UUID(meta["visitor_id"])

    def test_existing_conversation_meta_omits_known_ids(self, monkeypatch):
        """When conversation_id and visitor_id are supplied, meta omits them."""
        client = create_test_client(monkeypatch, _make_fake_pipelines())
        with client:
            resp = client.post(
                "/chat",
                json={
                    "question": "Q?",
                    "conversation_id": "11111111-1111-1111-1111-111111111111",
                    "visitor_id": "22222222-2222-2222-2222-222222222222",
                },
            )
        events = [
            json.loads(line.removeprefix("data: "))
            for line in resp.text.strip().split("\n\n")
            if line.startswith("data:")
        ]
        meta = next(e for e in events if e["type"] == "meta")
        assert "interaction_id" in meta
        assert "conversation_id" not in meta
        assert "visitor_id" not in meta

    def test_no_recording_without_deployment(self, monkeypatch):
        """Chat works normally when active_deployment_id is None (no recording)."""
        client = create_test_client(monkeypatch, _make_fake_pipelines())
        with client:
            resp = client.post("/chat", json={"question": "Q?"})
        assert resp.status_code == 200
        types = [
            json.loads(line.removeprefix("data: "))["type"]
            for line in resp.text.strip().split("\n\n")
            if line.startswith("data:")
        ]
        assert "meta" in types
        assert "done" in types
