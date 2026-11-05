from conftest import (
    FakeCitationBuilder,
    FakeContextBuilder,
    FakeStrategy,
    make_scored_chunk,
    make_streaming_result,
)

from geoai_rag.models import Answer
from geoai_rag.models.generation_result import GenerationResult
from geoai_rag.query import QueryPipeline


class FakeLLMClient:
    """Fake LLM that returns a canned answer."""

    def chat(self, messages, temperature=0.3, max_tokens=1024):
        """Return a canned answer."""
        return GenerationResult(text="U-Net is a convolutional neural network [1].")

    def chat_stream(self, messages, temperature=0.3, max_tokens=1024):
        """Stream the canned answer as a single token."""
        return make_streaming_result([self.chat(messages, temperature, max_tokens).text])


class FakeClock:
    """Deterministic stand-in for the ``time`` module inside the query pipeline.

    Time moves only when a fake collaborator calls ``advance``, so timing
    assertions are exact and never depend on real elapsed time.
    """

    def __init__(self) -> None:
        """Start the clock at zero seconds."""
        self._now = 0.0

    def monotonic(self) -> float:
        """Return the current fake time in seconds."""
        return self._now

    def advance(self, seconds: float) -> None:
        """Move the fake time forward by the given number of seconds."""
        self._now += seconds


class TimedFakeStrategy(FakeStrategy):
    """Retrieval strategy whose retrieve() consumes a fixed amount of fake time."""

    def __init__(self, results, clock: FakeClock, seconds: float) -> None:
        """Store the canned results and how long each retrieval takes."""
        super().__init__(results)
        self._clock = clock
        self._seconds = seconds

    def retrieve(self, question, top_k, filters=None):
        """Advance the fake clock, then return the canned results."""
        self._clock.advance(self._seconds)
        return super().retrieve(question, top_k, filters)


class TimedFakeLLMClient(FakeLLMClient):
    """LLM whose chat() consumes a fixed amount of fake time."""

    def __init__(self, clock: FakeClock, seconds: float) -> None:
        """Store how long each chat call takes."""
        self._clock = clock
        self._seconds = seconds

    def chat(self, messages, temperature=0.3, max_tokens=1024):
        """Advance the fake clock, then return the canned answer."""
        self._clock.advance(self._seconds)
        return super().chat(messages, temperature, max_tokens)


def test_ask_compose_chain():
    canned = [make_scored_chunk()]
    pipeline = QueryPipeline(
        strategy=FakeStrategy(canned),
        context_builder=FakeContextBuilder(),
        llm_client=FakeLLMClient(),
        citation_builder=FakeCitationBuilder(),
        top_k=8,
    )

    result = pipeline.ask("What is U-Net?")

    answer = result.answer
    assert answer.answer == "U-Net is a convolutional neural network [1]."
    assert len(answer.sources) == 1
    assert answer.sources[0].chunk.content == "U-Net architecture"
    assert len(answer.citations) == 1
    assert answer.citations[0].chapter == "03"
    assert answer.citations[0].section == "U-Net"


def test_ask_zero_chunks_returns_canned_answer():
    pipeline = QueryPipeline(
        strategy=FakeStrategy([]),
        context_builder=FakeContextBuilder(),
        llm_client=FakeLLMClient(),
        citation_builder=FakeCitationBuilder(),
        top_k=8,
    )

    result = pipeline.ask("What is something not in the book?")

    assert "No relevant information found" in result.answer.answer
    assert result.answer.sources == []
    assert result.answer.citations == []


def test_answer_timings_default_to_zero():
    """An Answer built without timings, as existing callers do, reports 0 for both."""
    answer = Answer(answer="A", sources=[], citations=[])

    assert answer.retrieval_ms == 0, "retrieval_ms must default to 0 for existing callers"
    assert answer.generation_ms == 0, "generation_ms must default to 0 for existing callers"


def test_answer_accepts_timings():
    """An Answer stores explicit retrieval and generation timings."""
    answer = Answer(answer="A", sources=[], citations=[], retrieval_ms=120, generation_ms=3400)

    assert answer.retrieval_ms == 120
    assert answer.generation_ms == 3400


def test_ask_reports_retrieval_and_generation_ms(monkeypatch):
    """Each stage timer wraps only its own call, so the two durations never mix."""
    clock = FakeClock()
    monkeypatch.setattr("geoai_rag.query.pipeline.time", clock)
    pipeline = QueryPipeline(
        strategy=TimedFakeStrategy([make_scored_chunk()], clock, seconds=0.25),
        context_builder=FakeContextBuilder(),
        llm_client=TimedFakeLLMClient(clock, seconds=1.5),
        citation_builder=FakeCitationBuilder(),
        top_k=8,
    )

    result = pipeline.ask("What is U-Net?")

    assert result.answer.retrieval_ms == 250, "retrieval_ms must time only the retrieve() call"
    assert result.answer.generation_ms == 1500, "generation_ms must time only the chat() call"


def test_ask_zero_chunks_reports_retrieval_ms_and_no_generation(monkeypatch):
    """The no-results early return keeps the real retrieval time and reports no generation."""
    clock = FakeClock()
    monkeypatch.setattr("geoai_rag.query.pipeline.time", clock)
    pipeline = QueryPipeline(
        strategy=TimedFakeStrategy([], clock, seconds=0.25),
        context_builder=FakeContextBuilder(),
        llm_client=TimedFakeLLMClient(clock, seconds=1.5),
        citation_builder=FakeCitationBuilder(),
        top_k=8,
    )

    result = pipeline.ask("What is something not in the book?")

    assert result.answer.retrieval_ms == 250, "zero-chunk path must keep the real retrieval time"
    assert result.answer.generation_ms == 0, "no LLM call happened, so generation_ms must be 0"
