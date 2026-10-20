"""Tests for the eval runner: YAML loading, question selection, and the scoring flow.

HTTP is faked at requests.get and requests.post, the judge LLM is a fake with the same chat
method as OpenAICompatibleClient, and the database is an in-memory stand-in for AsyncSession,
so no API server, LLM, or Postgres is needed.
The scorer functions are the real ones, so these tests also pin how the runner wires them up.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self, cast

import httpx
import openai
import pytest
import requests
from conftest import FakeCitationBuilder
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from geoai_rag.db.models import Deployment, EvalQuestion, EvalResult, EvalRun, EvalSet
from geoai_rag.eval.runner import (
    JUDGE_MAX_TOKENS,
    EvalRunner,
    EvalRunnerError,
    NoMatchingQuestionsError,
    load_eval_set,
    select_questions,
)
from geoai_rag.interfaces import LLMClient
from geoai_rag.models import Answer, Chunk, ScoredChunk

BASE_URL = "http://api.test"
ADMIN_TOKEN = "admin-secret"
DEPLOYMENT_ID = 7
CONFIG = {"retrieval_strategy": "hybrid", "retrieval_top_k": 8, "llm_model": "judge-model"}

# Source paths the API returns for the first question: one under an expected source, three not.
Q1_PATHS = [
    "book/03_Seg/01/a.ipynb",
    "book/05_TS/b.ipynb",
    "book/11_Ethics/c.md",
    "data/transcripts/x.json",
]

EVAL_YAML = """\
name: test-set
description: Runner test set
questions:
  - id: q001
    question: "What is U-Net?"
    expected_answer: |
      EXPECTED-ONE
    expected_sources:
      - "book/03_Seg"
      - "book/99_Missing"
    tags: ["chapter-3", "core"]
    notes: "not stored in the database"
  - id: q002
    question: "How is ethics applied?"
    expected_answer: "EXPECTED-TWO"
    expected_sources: ["book/11_Ethics"]
    tags: ["chapter-11"]
  - id: q003
    question: "Who is the book for?"
    expected_answer: "EXPECTED-THREE"
    expected_sources: ["book/index.md"]
"""


def question_yaml(question_id: str, text: str, sources: str = '["book/x"]') -> str:
    """Render one YAML question entry with the given id, text, and inline expected_sources."""
    return (
        f"  - id: {question_id}\n"
        f'    question: "{text}"\n'
        '    expected_answer: "answer"\n'
        f"    expected_sources: {sources}\n"
    )


class FakeClock:
    """Stand-in for the time module: time moves only when the fake API advances it."""

    def __init__(self) -> None:
        """Start the clock at zero seconds."""
        self.now = 0.0

    def monotonic(self) -> float:
        """Return the current fake time in seconds."""
        return self.now


class FakeResponse:
    """The parts of requests.Response that the runner reads."""

    def __init__(self, body: Any, status_code: int = 200) -> None:
        """Store the JSON body and derive ok and text from it."""
        self._body = body
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = json.dumps(body)

    def json(self) -> Any:
        """Return the canned JSON body."""
        return self._body


def default_judge(prompt: str) -> str:
    """Rate faithfulness prompts 0.6 and similarity prompts 0.8, and reject any other prompt."""
    if "CHUNK-TEXT" in prompt:
        return "0.6"
    if "EXPECTED-" in prompt:
        return "0.8"
    raise AssertionError(f"judge prompt has neither chunk text nor an expected answer: {prompt!r}")


class FakeHttp:
    """Stands in for requests.get and requests.post and records every call.

    GET /config answers with CONFIG.
    POST /ask answers from `answers`, keyed by question text, and takes 47 fake seconds.
    The judge LLM is not reached through requests at all, so any other request fails the test.
    """

    ASK_SECONDS = 47.0

    def __init__(self, clock: FakeClock) -> None:
        """Start with no canned answers."""
        self.clock = clock
        self.answers: dict[str, FakeResponse] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        """Answer GET /config; any other GET is a bug in the runner."""
        self.calls.append(("GET", url, kwargs))
        if url == f"{BASE_URL}/config":
            return FakeResponse(CONFIG)
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        """Answer POST /ask from the canned answers; any other POST is a bug in the runner."""
        self.calls.append(("POST", url, kwargs))
        if url == f"{BASE_URL}/ask":
            self.clock.now += self.ASK_SECONDS
            return self.answers[kwargs["json"]["question"]]
        raise AssertionError(f"unexpected POST {url}")


class FakeLlmClient:
    """Stands in for OpenAICompatibleClient and records every chat call.

    chat() answers with whatever `reply` returns for the prompt text.
    Its temperature and max_tokens defaults are the real client's, so a call that leaves them
    out is recorded with those defaults and the tests can see it.
    """

    def __init__(self) -> None:
        """Start with no recorded calls and the default judge."""
        self.calls: list[dict[str, Any]] = []
        self.reply: Callable[[str], str] = default_judge

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Record the call, then return what `reply` gives for the joined message texts."""
        prompt = "\n".join(message["content"] for message in messages)
        self.calls.append({"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens})
        return self.reply(prompt)


class FakeResult:
    """The slice of a SQLAlchemy Result that the runner calls."""

    def __init__(self, rows: list[Any]) -> None:
        """Hold the rows a SELECT found."""
        self._rows = rows

    def scalars(self) -> FakeResult:
        """Return self, because the runner only calls all() on the scalars."""
        return self

    def all(self) -> list[Any]:
        """Return every row."""
        return self._rows

    def scalar_one_or_none(self) -> Any:
        """Return the first row, or None when the SELECT found nothing."""
        return self._rows[0] if self._rows else None

    def scalar_one(self) -> Any:
        """Return the first row, raising NoResultFound like SQLAlchemy when there is none."""
        if not self._rows:
            raise NoResultFound("No row was found when one was required")
        return self._rows[0]


class FakeSession:
    """Just enough of AsyncSession for the runner: execute, get, add, flush, and commit.

    execute() ignores WHERE and ORDER BY clauses and returns every stored row of the selected
    class in the order it was stored, so a test seeds only the rows a query is meant to find.
    SQL predicates are therefore not covered by these tests.
    """

    def __init__(self, db: FakeDb) -> None:
        """Bind the session to the shared in-memory rows."""
        self._db = db
        self._pending: list[Any] = []

    async def __aenter__(self) -> Self:
        """Enter the async with block that scopes one unit of work."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Leave the block, dropping anything that was added but never flushed or committed."""
        self._pending.clear()

    async def execute(self, statement: Any) -> FakeResult:
        """Return every stored row of the ORM class that the SELECT reads."""
        entity = statement.column_descriptions[0]["entity"]
        return FakeResult(list(self._db.rows[entity]))

    async def get(self, entity: type, ident: int) -> Any:
        """Return the stored row with this primary key, or None."""
        return next((row for row in self._db.rows[entity] if row.id == ident), None)

    def add(self, obj: Any) -> None:
        """Queue obj for the next flush or commit."""
        self._pending.append(obj)

    async def flush(self) -> None:
        """Store the queued objects so they get their ids."""
        self._store_pending()

    async def commit(self) -> None:
        """Store the queued objects and end the transaction."""
        self._store_pending()

    def _store_pending(self) -> None:
        """Move the queued objects into the shared rows."""
        for obj in self._pending:
            self._db.store(obj)
        self._pending.clear()


class FakeDb:
    """In-memory rows keyed by ORM class, and the session factory the runner is given."""

    def __init__(self) -> None:
        """Start with no rows and no sessions opened."""
        self.rows: defaultdict[type, list[Any]] = defaultdict(list)
        self.sessions_opened = 0
        self._next_id: defaultdict[type, int] = defaultdict(lambda: 1)

    def __call__(self) -> FakeSession:
        """Open a session, the way calling an async_sessionmaker does."""
        self.sessions_opened += 1
        return FakeSession(self)

    def store(self, obj: Any) -> None:
        """Keep obj, giving it the next integer id the way an autoincrement key would."""
        kind = type(obj)
        if obj.id is None:
            obj.id = self._next_id[kind]
            self._next_id[kind] += 1
        if obj not in self.rows[kind]:
            self.rows[kind].append(obj)


def make_answer(answer: str, chunks: list[tuple[str, str]]) -> dict[str, Any]:
    """Build the JSON body POST /ask returns, from (source_path, content) pairs."""
    sources = [
        ScoredChunk(
            chunk=Chunk(
                id=f"chunk-{index}",
                content=content,
                content_hash=f"hash-{index}",
                source_type="book_text",
                content_type="concept",
                metadata={"source_path": path, "chapter": "03", "section": "Section"},
            ),
            score=3.0 - index,
            ranking_method="reranked",
        )
        for index, (path, content) in enumerate(chunks)
    ]
    return Answer(
        answer=answer,
        sources=sources,
        citations=FakeCitationBuilder().build(sources),
        retrieval_ms=120,
        generation_ms=3400,
    ).model_dump(mode="json")


def seed_answers(http: FakeHttp) -> None:
    """Give each eval question a canned /ask answer; only the first retrieves an expected source."""
    q1_chunks = [(path, f"CHUNK-TEXT-{index}") for index, path in enumerate(Q1_PATHS)]
    http.answers["What is U-Net?"] = FakeResponse(make_answer("ANSWER-TEXT-ONE", q1_chunks))
    http.answers["How is ethics applied?"] = FakeResponse(
        make_answer("ANSWER-TEXT-TWO", [("book/11_Ethics/index.qmd", "CHUNK-TEXT-ETHICS")])
    )
    http.answers["Who is the book for?"] = FakeResponse(
        make_answer("ANSWER-TEXT-THREE", [("book/index.md", "CHUNK-TEXT-INDEX")])
    )


@pytest.fixture
def eval_file(tmp_path: Path) -> Path:
    """Write the three-question eval set to a temporary file and return its path."""
    path = tmp_path / "test_set.yml"
    path.write_text(EVAL_YAML, encoding="utf-8")
    return path


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> FakeHttp:
    """Replace requests.get, requests.post, and the runner's clock with fakes."""
    clock = FakeClock()
    fake = FakeHttp(clock)
    monkeypatch.setattr(requests, "get", fake.get)
    monkeypatch.setattr(requests, "post", fake.post)
    monkeypatch.setattr("geoai_rag.eval.runner.time", clock)
    return fake


@pytest.fixture
def db() -> FakeDb:
    """An in-memory database holding the one active deployment a started API would have made."""
    fake = FakeDb()
    fake.rows[Deployment].append(
        Deployment(
            id=DEPLOYMENT_ID,
            prompt_version_id=1,
            index_run_id=1,
            model_name="judge-model",
            temperature=0.3,
            retrieval_strategy="hybrid",
            is_active=True,
        )
    )
    return fake


@pytest.fixture
def llm() -> FakeLlmClient:
    """A fake judge LLM that rates with default_judge until a test swaps in another reply."""
    return FakeLlmClient()


@pytest.fixture
def runner(db: FakeDb, llm: FakeLlmClient) -> EvalRunner:
    """A runner wired to the fake judge LLM and the fake database; HTTP comes from `http`."""
    return EvalRunner(
        base_url=BASE_URL,
        llm_client=cast("LLMClient", llm),
        session_factory=cast("async_sessionmaker[AsyncSession]", db),
        admin_token=ADMIN_TOKEN,
    )


def test_load_eval_set_returns_name_description_and_normalized_questions(eval_file):
    """The YAML becomes plain dicts, with whitespace stripped and missing tags defaulted."""
    data = load_eval_set(eval_file)

    assert data["name"] == "test-set"
    assert data["description"] == "Runner test set"
    assert [q["id"] for q in data["questions"]] == ["q001", "q002", "q003"]
    first = data["questions"][0]
    assert first["question"] == "What is U-Net?"
    assert first["expected_answer"] == "EXPECTED-ONE", "the block scalar's newline is stripped"
    assert first["expected_sources"] == ["book/03_Seg", "book/99_Missing"]
    assert first["tags"] == ["chapter-3", "core"]
    assert data["questions"][2]["tags"] == [], "a question without tags gets an empty list"


def test_load_eval_set_names_the_missing_file(tmp_path):
    """A wrong --eval-set path fails with the path in the message, not a bare traceback."""
    with pytest.raises(FileNotFoundError, match="nope.yml"):
        load_eval_set(tmp_path / "nope.yml")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        pytest.param(
            question_yaml("q001", "First?") + question_yaml("q001", "Second?"),
            "duplicate question id",
            id="duplicate-id",
        ),
        pytest.param(
            question_yaml("q001", "Same?") + question_yaml("q002", "Same?"),
            "duplicate question text",
            id="duplicate-text",
        ),
        pytest.param(
            question_yaml("q001", "First?", sources="[]"),
            "expected_sources",
            id="no-expected-sources",
        ),
    ],
)
def test_load_eval_set_rejects_content_that_would_break_a_run(tmp_path, body, message):
    """Repeated ids or texts and empty source lists fail at load, before any LLM time is spent."""
    path = tmp_path / "bad.yml"
    path.write_text(f"name: bad-set\nquestions:\n{body}", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_eval_set(path)


def test_select_questions_by_tag_keeps_questions_with_any_listed_tag(eval_file):
    """A question is selected when it carries at least one of the tags, and file order is kept."""
    selected = select_questions(load_eval_set(eval_file), tags=["core", "chapter-11"])

    assert [q["id"] for q in selected] == ["q001", "q002"]


@pytest.mark.parametrize(
    ("question_ids", "expected"),
    [
        pytest.param(["q003"], ["q003"], id="one-id"),
        pytest.param(["q003", "q001"], ["q001", "q003"], id="several-ids-in-file-order"),
        pytest.param(["q002", "q002"], ["q002"], id="repeated-id-selects-once"),
    ],
)
def test_select_questions_by_ids_returns_only_those_questions(eval_file, question_ids, expected):
    """--question picks the listed questions, each once, in the file's order and not as typed.

    A repeated id must not select its question twice, because a run stores one result per
    question.
    """
    selected = select_questions(load_eval_set(eval_file), question_ids=question_ids)

    assert [q["id"] for q in selected] == expected


def test_select_questions_applies_the_tag_filter_to_the_listed_ids(eval_file):
    """Ids and tags both narrow the run: a listed question without the tag is left out.

    That is not an error, because the id exists in the eval set; only a missing id is.
    """
    selected = select_questions(
        load_eval_set(eval_file), tags=["chapter-11"], question_ids=["q001", "q002"]
    )

    assert [q["id"] for q in selected] == ["q002"]


def test_select_questions_fails_when_any_listed_id_is_unknown(eval_file):
    """One mistyped id rejects the whole selection instead of quietly running fewer questions.

    The message names each id that is not in the eval set, and no id that is.
    """
    with pytest.raises(NoMatchingQuestionsError, match="test-set") as raised:
        select_questions(load_eval_set(eval_file), question_ids=["q999", "q001", "q998"])

    message = str(raised.value)
    assert "q998" in message and "q999" in message, "every unknown id is listed"
    assert "q001" not in message, "an id that exists is not reported as unmatched"


@pytest.mark.parametrize(
    ("question_ids", "tags"),
    [
        pytest.param(["q999"], None, id="unknown-id"),
        pytest.param(None, ["no-such-tag"], id="unknown-tag"),
        pytest.param(["q001"], ["chapter-11"], id="both-filters-must-pass"),
        pytest.param([], None, id="empty-id-list-selects-nothing"),
    ],
)
def test_select_questions_raises_when_nothing_matches(eval_file, question_ids, tags):
    """A typo in --tags or --question fails loudly instead of running an empty evaluation."""
    with pytest.raises(NoMatchingQuestionsError, match="test-set"):
        select_questions(load_eval_set(eval_file), tags=tags, question_ids=question_ids)


def test_dry_run_lists_the_selected_questions_and_touches_nothing(
    eval_file, http, db, llm, runner, capsys
):
    """--dry-run prints the filtered plan without calling the API, the judge, or the database."""
    summary = asyncio.run(runner.run(eval_file, tags=["chapter-11"], dry_run=True))

    out = capsys.readouterr().out
    assert "q002" in out
    assert "How is ethics applied?" in out
    assert "What is U-Net?" not in out and "Who is the book for?" not in out
    assert summary == {"eval_set": "test-set", "dry_run": True, "question_ids": ["q002"]}
    assert http.calls == [], "a dry run must not make any HTTP request"
    assert llm.calls == [], "a dry run must not call the judge LLM"
    assert db.sessions_opened == 0, "a dry run must not open a database session"


def test_sync_to_db_is_idempotent_and_updates_changed_questions(eval_file, db, runner):
    """Syncing twice reuses the rows, and an edited expected answer lands on the existing row."""
    data = load_eval_set(eval_file)

    first_set_id, first = asyncio.run(runner.sync_to_db(data))
    data["questions"][0]["expected_answer"] = "EXPECTED-ONE-REVISED"
    second_set_id, second = asyncio.run(runner.sync_to_db(data))

    assert second_set_id == first_set_id
    assert [q["db_id"] for q in second] == [q["db_id"] for q in first]
    assert len(db.rows[EvalSet]) == 1, "the eval set is found by name, not created again"
    assert len(db.rows[EvalQuestion]) == 3, "questions are matched by key, not duplicated"
    stored = {row.question_key: row for row in db.rows[EvalQuestion]}
    assert sorted(stored) == ["q001", "q002", "q003"], "each row stores its YAML id"
    assert stored["q001"].expected_answer == "EXPECTED-ONE-REVISED"
    assert stored["q001"].tags == ["chapter-3", "core"]


def test_sync_to_db_keeps_a_reworded_question_on_its_row(eval_file, db, runner):
    """Rewording a question updates its row, because rows are matched by YAML id, not by text."""
    data = load_eval_set(eval_file)
    _, first = asyncio.run(runner.sync_to_db(data))
    data["questions"][0]["question"] = "What is a U-Net?"

    _, second = asyncio.run(runner.sync_to_db(data))

    assert [q["db_id"] for q in second] == [q["db_id"] for q in first]
    assert len(db.rows[EvalQuestion]) == 3, "a reworded question must not add a row"
    reworded = next(row for row in db.rows[EvalQuestion] if row.question_key == "q001")
    assert reworded.question == "What is a U-Net?"


def test_sync_to_db_gives_rows_saved_without_a_key_their_key(eval_file, db, runner):
    """A row from before question_key existed is matched by its text and keyed, not duplicated."""
    eval_set = EvalSet(name="test-set", description="Runner test set")
    db.store(eval_set)
    unkeyed = EvalQuestion(
        eval_set_id=eval_set.id,
        question="What is U-Net?",
        expected_answer="stale",
        expected_sources=["book/old"],
        tags=None,
    )
    db.store(unkeyed)

    _, synced = asyncio.run(runner.sync_to_db(load_eval_set(eval_file)))

    assert len(db.rows[EvalQuestion]) == 3, "the unkeyed row is reused, not duplicated"
    assert synced[0]["db_id"] == unkeyed.id
    assert unkeyed.question_key == "q001"
    assert unkeyed.expected_answer == "EXPECTED-ONE", "the reused row is brought up to date"


def test_run_scores_and_stores_each_selected_question(eval_file, http, db, runner, capsys):
    """The selected question is asked, scored, and stored with exact scores, trace, snapshots."""
    seed_answers(http)

    summary = asyncio.run(runner.run(eval_file, question_ids=["q001"]))

    (run,) = db.rows[EvalRun]
    assert run.status == "completed"
    assert run.deployment_id == DEPLOYMENT_ID
    assert run.config == CONFIG
    assert len(db.rows[EvalQuestion]) == 3, "the whole YAML is synced even when one question runs"
    (result,) = db.rows[EvalResult]
    question_row = next(q for q in db.rows[EvalQuestion] if q.question_key == "q001")
    assert result.eval_run_id == run.id
    assert result.question_id == question_row.id
    assert result.question_snapshot == "What is U-Net?"
    assert result.expected_answer_snapshot == "EXPECTED-ONE"
    assert result.expected_sources_snapshot == ["book/03_Seg", "book/99_Missing"]
    assert result.response == "ANSWER-TEXT-ONE"
    assert [c["chunk_id"] for c in result.citations] == [f"chunk-{i}" for i in range(4)]
    expected_scores = {
        "retrieval_precision": 0.25,
        "retrieval_recall": 0.5,
        "answer_similarity": 0.8,
        "faithfulness": 0.6,
    }
    assert result.scores == expected_scores
    assert result.trace == {
        "round_trip_ms": 47000,
        "retrieval_ms": 120,
        "generation_ms": 3400,
        "chunks": [
            {
                "chunk_id": f"chunk-{index}",
                "source_path": path,
                "score": 3.0 - index,
                "ranking_method": "reranked",
            }
            for index, path in enumerate(Q1_PATHS)
        ],
        "judge_replies": {"answer_similarity": "0.8", "faithfulness": "0.6"},
    }
    assert summary["eval_run_id"] == run.id
    assert summary["status"] == "completed"
    assert summary["question_count"] == 1
    assert summary["averages"] == expected_scores
    assert "q001" in capsys.readouterr().out


def test_judge_calls_go_straight_to_the_llm_with_the_right_evidence(eval_file, http, llm, runner):
    """Judge prompts skip POST /ask; similarity sees the reference, faithfulness the chunks."""
    seed_answers(http)

    asyncio.run(runner.run(eval_file, question_ids=["q001"]))

    assert [(method, url) for method, url, _ in http.calls] == [
        ("GET", f"{BASE_URL}/config"),
        ("POST", f"{BASE_URL}/ask"),
    ]
    config_call, ask_call = http.calls
    assert config_call[2]["headers"] == {"Authorization": f"Bearer {ADMIN_TOKEN}"}
    assert ask_call[2]["json"] == {"question": "What is U-Net?"}
    assert len(llm.calls) == 2, "one similarity call and one faithfulness call"
    for call in llm.calls:
        assert call["temperature"] == 0.0, "judging must be repeatable, so sampling is off"
        assert call["max_tokens"] == JUDGE_MAX_TOKENS, "a reasoning model needs room to think"
    prompts = [call["prompt"] for call in llm.calls]
    with_reference = [p for p in prompts if "EXPECTED-ONE" in p]
    with_chunks = [p for p in prompts if "CHUNK-TEXT-0" in p]
    assert len(with_reference) == 1 and len(with_chunks) == 1
    assert "ANSWER-TEXT-ONE" in with_reference[0] and "CHUNK-TEXT" not in with_reference[0]
    assert "ANSWER-TEXT-ONE" in with_chunks[0] and "CHUNK-TEXT-3" in with_chunks[0]
    assert "EXPECTED-" not in with_chunks[0], "faithfulness is judged on the chunks alone"


def test_api_error_skips_question_and_completes_with_partial_results(eval_file, http, db, runner):
    """A failing question is retried then skipped; the run completes with partial results."""
    seed_answers(http)
    http.answers["How is ethics applied?"] = FakeResponse(
        {"detail": "generation failed"}, status_code=503
    )

    asyncio.run(runner.run(eval_file))

    (run,) = db.rows[EvalRun]
    assert run.status == "completed"
    scored_questions = [r.question_snapshot for r in db.rows[EvalResult]]
    assert "What is U-Net?" in scored_questions
    assert "How is ethics applied?" not in scored_questions


def test_run_fails_before_asking_anything_when_no_deployment_is_active(eval_file, http, db, runner):
    """Results need a deployment to attach to, so the run stops before spending any LLM time."""
    seed_answers(http)
    db.rows[Deployment].clear()

    with pytest.raises(EvalRunnerError, match="active deployment"):
        asyncio.run(runner.run(eval_file))

    assert not any(url.endswith("/ask") for _, url, _ in http.calls)
    assert db.rows[EvalRun] == []


def test_start_run_creates_the_run_row_and_asks_nothing(eval_file, http, db, llm, runner):
    """start_run returns the run id with the row 'running' before any question is asked.

    The API answers POST /eval/runs with this id and scores the questions afterwards.
    """
    started = asyncio.run(runner.start_run(eval_file, tags=["chapter-11"]))

    (run,) = db.rows[EvalRun]
    assert run.status == "running"
    assert started.run_id == run.id
    assert started.eval_set_name == "test-set"
    assert started.set_size == 3, "the whole eval set is counted, not only the selected questions"
    assert [q["id"] for q in started.questions] == ["q002"]
    question_row = next(q for q in db.rows[EvalQuestion] if q.question_key == "q002")
    assert started.questions[0]["db_id"] == question_row.id
    assert [(method, url) for method, url, _ in http.calls] == [("GET", f"{BASE_URL}/config")]
    assert llm.calls == [], "starting a run must not call the judge"
    assert db.rows[EvalResult] == [], "no question has been scored yet"


def test_execute_run_finishes_the_run_that_start_run_created(eval_file, http, db, runner):
    """execute_run asks only the started questions and completes the same run row."""
    seed_answers(http)
    started = asyncio.run(runner.start_run(eval_file, question_ids=["q002", "q003"]))

    summary = asyncio.run(runner.execute_run(started))

    (run,) = db.rows[EvalRun]
    assert run.id == started.run_id
    assert run.status == "completed"
    assert [r.question_snapshot for r in db.rows[EvalResult]] == [
        "How is ethics applied?",
        "Who is the book for?",
    ]
    assert summary["eval_run_id"] == started.run_id
    assert summary["question_count"] == 2


def test_start_run_rejects_unmatched_filters_before_any_side_effect(eval_file, http, db, runner):
    """A filter typo fails before any row is written or request is sent.

    The API reports this to the caller at once, so it must not leave a run row behind.
    A mistyped id among valid ones counts too: the valid question must not start a run either.
    """
    with pytest.raises(NoMatchingQuestionsError, match="q999"):
        asyncio.run(runner.start_run(eval_file, question_ids=["q001", "q999"]))

    assert db.sessions_opened == 0
    assert http.calls == []


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        pytest.param(
            RuntimeError("LLM returned empty content"),
            "reasoning model",
            id="empty-reply",
        ),
        pytest.param(
            openai.APIConnectionError(
                request=httpx.Request("POST", "http://llm.test/v1/chat/completions")
            ),
            "APIConnectionError",
            id="endpoint-unreachable",
        ),
    ],
)
def test_judge_llm_turns_client_failures_into_runner_errors(runner, llm, failure, message):
    """A failed or empty judge call stops the run with EvalRunnerError, never a 0.0 score.

    main() reports only EvalRunnerError as a one-line error, so a raw client error would
    reach the user as a traceback.
    """

    def fail(prompt: str) -> str:
        """Raise the failure the real client would raise."""
        raise failure

    llm.reply = fail

    with pytest.raises(EvalRunnerError, match=message) as raised:
        runner.judge_llm([{"role": "user", "content": "rate this"}])

    assert raised.value.__cause__ is failure, "the client's own error stays attached"


def test_ask_question_rejects_a_body_that_is_not_an_answer(http, runner):
    """If /ask drifts from the Answer model the run stops with a clear error, not a KeyError."""
    http.answers["What is U-Net?"] = FakeResponse({"unexpected": True})

    with pytest.raises(EvalRunnerError, match="Answer"):
        runner.ask_question("What is U-Net?")
