from conftest import (
    FakeCitationBuilder,
    FakeContextBuilder,
    FakeStrategy,
    make_scored_chunk,
    make_streaming_result,
)

from geoai_rag.models.generation_result import GenerationResult
from geoai_rag.query import QueryPipeline


class FakeStreamingLLMClient:
    """Fake LLM that yields multiple tokens."""

    def chat(self, messages, temperature=0.3, max_tokens=1024):
        """Return a canned answer."""
        return GenerationResult(text="U-Net is a CNN [1].")

    def chat_stream(self, messages, temperature=0.3, max_tokens=1024):
        """Yield answer as three separate tokens."""
        return make_streaming_result(["U-Net ", "is ", "a CNN [1]."])


class FakeEmptyStreamLLMClient:
    """Fake LLM that yields only empty/whitespace tokens."""

    def chat(self, messages, temperature=0.3, max_tokens=1024):
        """Return an empty answer."""
        return GenerationResult(text="")

    def chat_stream(self, messages, temperature=0.3, max_tokens=1024):
        """Yield empty and whitespace-only tokens."""
        return make_streaming_result(["", "  "])


class TestAskStream:
    def test_yields_meta_then_tokens_then_done(self):
        pipeline = QueryPipeline(
            strategy=FakeStrategy([make_scored_chunk()]),
            context_builder=FakeContextBuilder(),
            llm_client=FakeStreamingLLMClient(),
            citation_builder=FakeCitationBuilder(),
            top_k=8,
        )
        events = list(pipeline.ask_stream("What is U-Net?"))
        types = [e["type"] for e in events]
        assert types[0] == "meta"
        assert types[-1] == "done"
        assert "token" in types

    def test_meta_has_citations(self):
        pipeline = QueryPipeline(
            strategy=FakeStrategy([make_scored_chunk()]),
            context_builder=FakeContextBuilder(),
            llm_client=FakeStreamingLLMClient(),
            citation_builder=FakeCitationBuilder(),
            top_k=8,
        )
        events = list(pipeline.ask_stream("What is U-Net?"))
        meta = events[0]
        assert meta["type"] == "meta"
        assert len(meta["citations"]) == 1
        assert meta["citations"][0]["url"] == "/ch1.html"
        assert meta["citations"][0]["display_label"]

    def test_zero_chunks_emits_meta_then_canned(self):
        pipeline = QueryPipeline(
            strategy=FakeStrategy([]),
            context_builder=FakeContextBuilder(),
            llm_client=FakeStreamingLLMClient(),
            citation_builder=FakeCitationBuilder(),
            top_k=8,
        )
        events = list(pipeline.ask_stream("Unknown topic?"))
        types = [e["type"] for e in events]
        assert types == ["meta", "token", "done"]
        assert events[0]["citations"] == []
        assert "No relevant information" in events[1]["content"]

    def test_empty_stream_yields_error(self):
        pipeline = QueryPipeline(
            strategy=FakeStrategy([make_scored_chunk()]),
            context_builder=FakeContextBuilder(),
            llm_client=FakeEmptyStreamLLMClient(),
            citation_builder=FakeCitationBuilder(),
            top_k=8,
        )
        events = list(pipeline.ask_stream("Q?"))
        types = [e["type"] for e in events]
        assert "error" in types
        assert "done" not in types

    def test_recording_ctx_populated(self):
        """_recording_ctx dict is populated with chunk data and timing."""
        pipeline = QueryPipeline(
            strategy=FakeStrategy([make_scored_chunk()]),
            context_builder=FakeContextBuilder(),
            llm_client=FakeStreamingLLMClient(),
            citation_builder=FakeCitationBuilder(),
            top_k=8,
        )
        recording_ctx = {}
        events = list(pipeline.ask_stream("What is U-Net?", _recording_ctx=recording_ctx))
        # Events unchanged
        types = [e["type"] for e in events]
        assert types[0] == "meta"
        assert types[-1] == "done"
        # Recording context populated
        assert len(recording_ctx["scored_chunks"]) == 1
        assert recording_ctx["scored_chunks"][0]["chunk_id"]
        assert recording_ctx["scored_chunks"][0]["source_type"] == "book_text"
        assert recording_ctx["retrieval_query"] == "What is U-Net?"
        assert isinstance(recording_ctx["retrieval_ms"], int)
        assert isinstance(recording_ctx["generation_wall_ms"], int)

    def test_recording_ctx_zero_results(self):
        """_recording_ctx is populated with empty chunks on zero-results path."""
        pipeline = QueryPipeline(
            strategy=FakeStrategy([]),
            context_builder=FakeContextBuilder(),
            llm_client=FakeStreamingLLMClient(),
            citation_builder=FakeCitationBuilder(),
            top_k=8,
        )
        recording_ctx = {}
        events = list(pipeline.ask_stream("Unknown?", _recording_ctx=recording_ctx))
        types = [e["type"] for e in events]
        assert types == ["meta", "token", "done"]
        assert recording_ctx["scored_chunks"] == []
        assert recording_ctx["retrieval_query"] == "Unknown?"
        assert recording_ctx["retrieval_ms"] >= 0
        assert recording_ctx["generation_wall_ms"] == 0


class TestOpenAIStreamingAdapter:
    def test_chat_stream_yields_content_skips_empty_deltas(self, monkeypatch):
        """Verify the real adapter calls stream=True and filters empty deltas."""
        from unittest.mock import MagicMock

        from geoai_rag.generation.llm_client import OpenAICompatibleClient

        chunk_with_content = MagicMock()
        chunk_with_content.choices = [MagicMock()]
        chunk_with_content.choices[0].delta.content = "Hello"
        chunk_with_content.choices[0].finish_reason = None

        chunk_empty_delta = MagicMock()
        chunk_empty_delta.choices = [MagicMock()]
        chunk_empty_delta.choices[0].finish_reason = None
        chunk_empty_delta.choices[0].delta.content = None

        chunk_no_choices = MagicMock()
        chunk_no_choices.choices = []

        chunk_final = MagicMock()
        chunk_final.choices = [MagicMock()]
        chunk_final.choices[0].finish_reason = None
        chunk_final.choices[0].delta.content = " world"

        # MagicMock attributes are truthy and non-primitive, which would fail
        # GenerationResult validation, so pin usage and model explicitly.
        for c in (chunk_with_content, chunk_empty_delta, chunk_no_choices, chunk_final):
            c.usage = None
            c.model = "test-model"

        fake_response = [chunk_with_content, chunk_empty_delta, chunk_no_choices, chunk_final]

        mock_openai = MagicMock()
        mock_openai.return_value.chat.completions.create.return_value = iter(fake_response)
        monkeypatch.setattr("openai.OpenAI", mock_openai)

        client = OpenAICompatibleClient(base_url="http://fake", api_key="fake", model="test-model")
        tokens = list(client.chat_stream([{"role": "user", "content": "Hi"}]))

        assert tokens == ["Hello", " world"]
        call_kwargs = mock_openai.return_value.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("stream") is True or call_kwargs[1].get("stream") is True


class TestBuildRetrievalQuery:
    def test_no_history(self):
        assert QueryPipeline._build_retrieval_query("What is X?", None) == "What is X?"

    def test_with_history(self):
        history = [
            {"role": "user", "content": "What is U-Net?"},
            {"role": "assistant", "content": "A CNN."},
        ]
        result = QueryPipeline._build_retrieval_query("Tell me more", history)
        assert result.startswith("Tell me more")
        assert "U-Net" in result

    def test_caps_at_500_chars(self):
        history = [{"role": "user", "content": "x" * 1000}]
        result = QueryPipeline._build_retrieval_query("Q", history)
        assert len(result) <= 504  # "Q " + 500 + margin

    def test_malformed_history(self):
        history = [{"bad_key": "value"}]
        result = QueryPipeline._build_retrieval_query("Q?", history)
        assert result == "Q?"
