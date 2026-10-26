"""Tests for the /eval endpoints: reading runs and eval sets, and starting a run.

The database is a canned FakeSession and the runner is a FakeRunner, so no PostgreSQL, LLM, or
running API server is needed.
With a canned session the tests show what each endpoint returns and which table and ordering its
statement uses; they cannot show that the SQL is valid, which needs a real PostgreSQL.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import create_test_client
from pydantic import SecretStr

from api.dependencies import Pipelines, require_admin_token, require_db_session
from api.routes import eval as eval_routes
from geoai_rag.config import Settings
from geoai_rag.db.models import EvalQuestion, EvalResult, EvalRun, EvalSet, VEvalRunSummary
from geoai_rag.eval.runner import (
    JUDGE_TIMEOUT_SECONDS,
    EvalRunnerError,
    NoMatchingQuestionsError,
    StartedRun,
)

RUN_ID = 41
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
# Stands in for the app's async_sessionmaker, which the endpoint only hands on to the runner.
SESSION_FACTORY = object()

SCORES = {
    "retrieval_precision": 0.5,
    "retrieval_recall": 1.0,
    "answer_similarity": 0.8,
    "faithfulness": 0.7,
}
TRACE = {"round_trip_ms": 47000, "chunks": []}

_ROUTES = [
    ("GET", "/eval/runs"),
    ("GET", "/eval/runs/1"),
    ("GET", "/eval/sets"),
    ("GET", "/eval/sets/1/questions"),
    ("POST", "/eval/runs"),
]


class FakeResult:
    """The slice of a SQLAlchemy Result that the eval endpoints call: scalars().all() and all()."""

    def __init__(self, rows: list[Any]) -> None:
        """Hold the rows a statement found."""
        self._rows = rows

    def scalars(self) -> FakeResult:
        """Return self, because the endpoints only call all() on the scalars."""
        return self

    def all(self) -> list[Any]:
        """Return every row."""
        return self._rows


class FakeSession:
    """Canned stand-in for AsyncSession.

    execute() answers with the queued rows in order and keeps every statement it was given.
    get() answers from the stored rows by class and primary key, and keeps every lookup.
    """

    def __init__(self) -> None:
        """Start with nothing queued, stored, or recorded."""
        self.statements: list[Any] = []
        self.gets: list[tuple[type, int]] = []
        self._queued: list[list[Any]] = []
        self._stored: dict[tuple[type, int], Any] = {}

    def queue(self, rows: list[Any]) -> None:
        """Queue the rows that the next unanswered execute() returns."""
        self._queued.append(rows)

    def store(self, row: Any) -> None:
        """Make get() return row for its class and id."""
        self._stored[(type(row), row.id)] = row

    async def execute(self, statement: Any) -> FakeResult:
        """Record the statement and return the next queued rows."""
        self.statements.append(statement)
        return FakeResult(self._queued.pop(0))

    async def get(self, entity: type, ident: int) -> Any:
        """Record the lookup and return the stored row, or None."""
        self.gets.append((entity, ident))
        return self._stored.get((entity, ident))


class FakeRunner:
    """Stand-in for EvalRunner that records what the endpoint asks of it instead of running."""

    def __init__(self) -> None:
        """Start with no calls recorded and a runner whose start and execution succeed."""
        self.started_with: list[tuple[Path, list[str] | None, list[str] | None]] = []
        self.executed: list[StartedRun] = []
        self.start_error: Exception | None = None
        self.execute_error: Exception | None = None

    async def start_run(
        self,
        yaml_path: Path,
        tags: list[str] | None = None,
        question_ids: list[str] | None = None,
        source: str = "cli",
    ) -> StartedRun:
        """Record the request, then fail with start_error or return a run with id RUN_ID."""
        self.started_with.append((yaml_path, tags, question_ids))
        if self.start_error is not None:
            raise self.start_error
        return StartedRun(run_id=RUN_ID, eval_set_name="golden-qa-v1", set_size=3, questions=[])

    async def execute_run(self, started: StartedRun) -> dict[str, Any]:
        """Record the run, then fail with execute_error or return a minimal summary."""
        self.executed.append(started)
        if self.execute_error is not None:
            raise self.execute_error
        return {"eval_run_id": started.run_id, "status": "completed"}


def _bypass_admin_token(monkeypatch):
    """Override the admin auth dependency so data tests skip auth."""
    from api.main import app

    async def no_admin_check() -> None:
        """Stand-in for require_admin_token."""

    monkeypatch.setitem(app.dependency_overrides, require_admin_token, no_admin_check)


def _app_state():
    """Return the app's State, where POST /eval/runs keeps its mark that a run is active."""
    from api.main import app

    return app.state


@pytest.fixture
def session() -> FakeSession:
    """The canned session that every GET endpoint reads through."""
    return FakeSession()


@pytest.fixture
def client(monkeypatch, session):
    """TestClient with admin auth bypassed, the canned session, and a session factory set."""
    test_client = create_test_client(monkeypatch, Pipelines())
    from api.main import app

    _bypass_admin_token(monkeypatch)

    async def canned_session() -> FakeSession:
        """Stand-in for require_db_session so no real database is needed."""
        return session

    monkeypatch.setitem(app.dependency_overrides, require_db_session, canned_session)
    with test_client:
        # Without DATABASE_URL the lifespan leaves the factory unset, but POST reads it.
        monkeypatch.setattr(app.state, "db_session_factory", SESSION_FACTORY)
        eval_routes._release_run_slot(app.state)
        yield test_client
        eval_routes._release_run_slot(app.state)


@pytest.fixture
def client_no_db(monkeypatch):
    """TestClient with admin auth bypassed but the database disabled."""
    _bypass_admin_token(monkeypatch)
    with create_test_client(monkeypatch, Pipelines()) as test_client:
        yield test_client


@pytest.fixture
def auth_client(monkeypatch):
    """TestClient with the real admin auth dependency (no override)."""
    monkeypatch.setenv("ADMIN_TOKEN", "correct-token")
    with create_test_client(monkeypatch, Pipelines()) as test_client:
        yield test_client


@pytest.fixture
def runner(monkeypatch) -> FakeRunner:
    """The fake runner that POST /eval/runs gets from the endpoint's runner builder."""
    fake = FakeRunner()
    monkeypatch.setattr(eval_routes, "_build_runner", lambda settings, session_factory: fake)
    return fake


@pytest.fixture
def eval_dir(monkeypatch, tmp_path) -> Path:
    """An empty directory that stands in for data/eval during the test."""
    monkeypatch.setattr(eval_routes, "_eval_set_dir", lambda: tmp_path)
    return tmp_path


def write_eval_set(directory: Path, file_name: str, set_name: str) -> Path:
    """Write a one-question eval set called set_name into directory and return its path."""
    path = directory / file_name
    path.write_text(
        f"name: {set_name}\n"
        "questions:\n"
        "  - id: q001\n"
        '    question: "What is U-Net?"\n'
        '    expected_answer: "A patch-based encoder-decoder network."\n'
        '    expected_sources: ["book/03_Semantic_Segmentation"]\n',
        encoding="utf-8",
    )
    return path


def make_result(result_id: int, question: str, question_id: int | None) -> EvalResult:
    """Build a stored result of run 7 with fixed scores and trace."""
    return EvalResult(
        id=result_id,
        eval_run_id=7,
        question_id=question_id,
        question_snapshot=question,
        expected_answer_snapshot=f"Expected: {question}",
        expected_sources_snapshot=["book/03_Semantic_Segmentation"],
        response=f"Answer: {question}",
        citations=[],
        trace=TRACE,
        scores=SCORES,
    )


# --- GET /eval/runs ---


def test_runs_returns_summaries_newest_first(client, session):
    """A run without results has null averages; the query reads the view newest first."""
    older = NOW - timedelta(days=1)
    session.queue(
        [
            VEvalRunSummary(
                eval_run_id=2,
                eval_set_id=1,
                deployment_id=3,
                status="completed",
                source="cli",
                created_at=NOW,
                deployment_label="v1",
                questions_evaluated=15,
                avg_retrieval_precision=0.45,
                avg_retrieval_recall=0.72,
                avg_answer_similarity=0.68,
                avg_faithfulness=0.81,
            ),
            VEvalRunSummary(
                eval_run_id=1,
                eval_set_id=1,
                deployment_id=3,
                status="running",
                source="api",
                created_at=older,
                deployment_label=None,
                questions_evaluated=0,
                avg_retrieval_precision=None,
                avg_retrieval_recall=None,
                avg_answer_similarity=None,
                avg_faithfulness=None,
            ),
        ]
    )

    resp = client.get("/eval/runs")

    assert resp.status_code == 200
    first, second = resp.json()
    assert datetime.fromisoformat(first.pop("created_at")) == NOW
    assert datetime.fromisoformat(second.pop("created_at")) == older
    assert first == {
        "eval_run_id": 2,
        "eval_set_id": 1,
        "deployment_id": 3,
        "status": "completed",
        "source": "cli",
        "planned_count": None,
        "last_heartbeat_at": None,
        "deployment_label": "v1",
        "questions_evaluated": 15,
        "avg_retrieval_precision": 0.45,
        "avg_retrieval_recall": 0.72,
        "avg_answer_similarity": 0.68,
        "avg_faithfulness": 0.81,
    }
    assert second == {
        "eval_run_id": 1,
        "eval_set_id": 1,
        "deployment_id": 3,
        "status": "running",
        "source": "api",
        "planned_count": None,
        "last_heartbeat_at": None,
        "deployment_label": None,
        "questions_evaluated": 0,
        "avg_retrieval_precision": None,
        "avg_retrieval_recall": None,
        "avg_answer_similarity": None,
        "avg_faithfulness": None,
    }
    sql = str(session.statements[0])
    assert "FROM geoai.v_eval_run_summary" in sql
    assert "ORDER BY geoai.v_eval_run_summary.created_at DESC" in sql


def test_runs_limit_defaults_to_50_and_accepts_a_caller_value(client, session):
    """The row cap bound into the query is the documented default, or the caller's when given."""
    session.queue([])
    session.queue([])

    assert client.get("/eval/runs").status_code == 200
    assert client.get("/eval/runs", params={"limit": 5}).status_code == 200

    bound = [list(statement.compile().params.values()) for statement in session.statements]
    assert bound == [[50], [5]]


# --- GET /eval/runs/{run_id} ---


def test_run_detail_returns_each_result_with_its_question_key(client, session):
    """Text and expected answer come from the result's snapshots; the key from the question row.

    A result whose question row was deleted is still listed, with a null key.
    """
    session.store(
        EvalRun(
            id=7,
            eval_set_id=1,
            deployment_id=3,
            status="completed",
            config={"retrieval_strategy": "hybrid"},
            created_at=NOW,
        )
    )
    session.queue(
        [
            (make_result(70, "What is semantic segmentation?", question_id=5), "q001"),
            (make_result(71, "A question that was deleted later", question_id=None), None),
        ]
    )

    resp = client.get("/eval/runs/7")

    assert resp.status_code == 200
    body = resp.json()
    assert datetime.fromisoformat(body.pop("created_at")) == NOW
    assert body == {
        "eval_run_id": 7,
        "eval_set_id": 1,
        "status": "completed",
        "config": {"retrieval_strategy": "hybrid"},
        "results": [
            {
                "question": "What is semantic segmentation?",
                "question_key": "q001",
                "expected_answer": "Expected: What is semantic segmentation?",
                "response": "Answer: What is semantic segmentation?",
                "scores": SCORES,
                "trace": TRACE,
            },
            {
                "question": "A question that was deleted later",
                "question_key": None,
                "expected_answer": "Expected: A question that was deleted later",
                "response": "Answer: A question that was deleted later",
                "scores": SCORES,
                "trace": TRACE,
            },
        ],
    }
    statement = session.statements[0]
    assert "LEFT OUTER JOIN geoai.eval_questions" in str(statement)
    assert 7 in statement.compile().params.values(), "only this run's results are selected"


def test_run_detail_for_an_unknown_run_is_404(client, session):
    """An unknown run id is a 404, and no results are read for it."""
    resp = client.get("/eval/runs/99")

    assert resp.status_code == 404
    assert resp.json() == {"detail": "Eval run not found"}
    assert session.gets == [(EvalRun, 99)]
    assert session.statements == []


# --- GET /eval/sets ---


def test_sets_returns_each_set_with_its_question_count(client, session):
    """An empty set is listed with a count of 0, and a set may have no description."""
    session.queue(
        [
            (
                EvalSet(id=1, name="golden-qa-v1", description="Core Q&A set", created_at=NOW),
                15,
            ),
            (EvalSet(id=2, name="scratch", description=None, created_at=NOW), 0),
        ]
    )

    resp = client.get("/eval/sets")

    assert resp.status_code == 200
    first, second = resp.json()
    assert datetime.fromisoformat(first.pop("created_at")) == NOW
    assert datetime.fromisoformat(second.pop("created_at")) == NOW
    assert first == {
        "id": 1,
        "name": "golden-qa-v1",
        "description": "Core Q&A set",
        "question_count": 15,
    }
    assert second == {"id": 2, "name": "scratch", "description": None, "question_count": 0}
    sql = str(session.statements[0])
    assert "LEFT OUTER JOIN geoai.eval_questions" in sql
    assert "GROUP BY geoai.eval_sets.id" in sql


# --- GET /eval/sets/{set_id}/questions ---


def test_set_questions_returns_questions_with_missing_tags_as_empty(client, session):
    """A question saved without a key or tags has a null key and an empty tag list."""
    session.store(EvalSet(id=1, name="golden-qa-v1", description=None, created_at=NOW))
    session.queue(
        [
            EvalQuestion(
                id=1,
                eval_set_id=1,
                question_key="q001",
                question="What is semantic segmentation?",
                expected_answer="A class label for every pixel.",
                expected_sources=["book/03_Semantic_Segmentation"],
                tags=["chapter-3", "core-concept"],
            ),
            EvalQuestion(
                id=2,
                eval_set_id=1,
                question_key=None,
                question="An older question",
                expected_answer=None,
                expected_sources=["book/11_Ethics"],
                tags=None,
            ),
        ]
    )

    resp = client.get("/eval/sets/1/questions")

    assert resp.status_code == 200
    assert resp.json() == [
        {
            "id": 1,
            "question_key": "q001",
            "question": "What is semantic segmentation?",
            "expected_answer": "A class label for every pixel.",
            "expected_sources": ["book/03_Semantic_Segmentation"],
            "tags": ["chapter-3", "core-concept"],
        },
        {
            "id": 2,
            "question_key": None,
            "question": "An older question",
            "expected_answer": None,
            "expected_sources": ["book/11_Ethics"],
            "tags": [],
        },
    ]
    assert session.gets == [(EvalSet, 1)]
    statement = session.statements[0]
    assert 1 in statement.compile().params.values(), "only this set's questions are selected"
    assert "ORDER BY geoai.eval_questions.id" in str(statement)


def test_set_questions_for_an_unknown_set_is_404(client, session):
    """An unknown set id is a 404, and no questions are read for it."""
    resp = client.get("/eval/sets/99/questions")

    assert resp.status_code == 404
    assert resp.json() == {"detail": "Eval set not found"}
    assert session.statements == []


@pytest.mark.parametrize("path", ["/eval/runs", "/eval/sets", "/eval/sets/1/questions"])
def test_list_endpoints_return_an_empty_list_when_there_are_no_rows(client, session, path):
    """No rows is a normal empty result, not an error."""
    session.store(EvalSet(id=1, name="empty", description=None, created_at=NOW))
    session.queue([])

    resp = client.get(path)

    assert resp.status_code == 200
    assert resp.json() == []


# --- POST /eval/runs ---


def test_post_starts_the_named_eval_set_and_scores_it_in_the_background(client, runner, eval_dir):
    """The set is found by the name inside its file, filters pass through, and the reply is 202.

    The reply carries the id of the run that start_run created, and that same run is executed.
    Several question ids reach start_run as the one list the caller sent.
    """
    write_eval_set(eval_dir, "a_other.yml", "other-set")
    golden = write_eval_set(eval_dir, "b_golden.yml", "golden-qa-v1")

    resp = client.post(
        "/eval/runs",
        json={
            "eval_set_name": "golden-qa-v1",
            "tags": ["chapter-3"],
            "question_ids": ["q001", "q003"],
        },
    )

    assert resp.status_code == 202
    assert resp.json() == {"eval_run_id": RUN_ID, "status": "running"}
    assert runner.started_with == [(golden, ["chapter-3"], ["q001", "q003"])]
    assert [started.run_id for started in runner.executed] == [RUN_ID]


def test_post_without_a_body_runs_the_first_eval_set_file(client, runner, eval_dir):
    """Naming no eval set picks the first file in file name order, with no filters."""
    first = write_eval_set(eval_dir, "a_first.yml", "set-a")
    write_eval_set(eval_dir, "b_second.yml", "set-b")

    resp = client.post("/eval/runs")

    assert resp.status_code == 202
    assert runner.started_with == [(first, None, None)]


def test_post_for_an_unknown_eval_set_name_is_404(client, runner, eval_dir):
    """A name that no eval set file declares is refused before any run is started."""
    write_eval_set(eval_dir, "golden.yml", "golden-qa-v1")

    resp = client.post("/eval/runs", json={"eval_set_name": "no-such-set"})

    assert resp.status_code == 404
    assert resp.json() == {"detail": "Eval set 'no-such-set' not found"}
    assert runner.started_with == []


def test_post_is_404_when_there_are_no_eval_set_files(client, runner, eval_dir):
    """A server without eval set files, such as the Docker image, cannot start a run."""
    resp = client.post("/eval/runs")

    assert resp.status_code == 404
    assert resp.json() == {"detail": "No eval set files found"}
    assert runner.started_with == []


def test_post_rejects_unknown_fields_instead_of_running_the_default_set(client, runner, eval_dir):
    """A misspelled field, such as eval_set, must not silently run the default eval set."""
    write_eval_set(eval_dir, "golden.yml", "golden-qa-v1")

    resp = client.post("/eval/runs", json={"eval_set": "golden-qa-v1"})

    assert resp.status_code == 422
    assert runner.started_with == []


def test_post_rejects_an_empty_question_ids_list_instead_of_running_every_question(
    client, runner, eval_dir
):
    """An empty list is a mistake, not "all questions": the field is left out for that."""
    write_eval_set(eval_dir, "golden.yml", "golden-qa-v1")

    resp = client.post("/eval/runs", json={"question_ids": []})

    assert resp.status_code == 422
    assert runner.started_with == []


@pytest.mark.parametrize(
    ("failure", "status", "detail"),
    [
        pytest.param(
            NoMatchingQuestionsError("No question in 'golden-qa-v1' matches tags nope"),
            422,
            "No question in 'golden-qa-v1' matches tags nope",
            id="filters-select-no-question",
        ),
        pytest.param(
            EvalRunnerError("No active deployment in the database."),
            503,
            "No active deployment in the database.",
            id="runner-precondition-fails",
        ),
        pytest.param(
            ValueError("/srv/data/eval/golden.yml is not a valid eval set"),
            500,
            "Eval set file is invalid; see the server log",
            id="eval-set-file-is-invalid",
        ),
    ],
)
def test_post_explains_why_a_run_could_not_start_and_frees_the_slot(
    client, runner, eval_dir, failure, status, detail
):
    """Each expected start failure is a clear HTTP error, starts no run, and blocks nothing later.

    The invalid-file case must not leak the server's file path to the caller.
    """
    write_eval_set(eval_dir, "golden.yml", "golden-qa-v1")
    runner.start_error = failure

    resp = client.post("/eval/runs")

    assert resp.status_code == status
    assert resp.json() == {"detail": detail}
    assert runner.executed == []
    runner.start_error = None
    assert client.post("/eval/runs").status_code == 202, "a failed start must free the slot"


def test_post_while_another_run_is_active_is_409(client, runner, eval_dir):
    """Runs are not stacked: a start while one is active is refused and starts nothing."""
    write_eval_set(eval_dir, "golden.yml", "golden-qa-v1")
    other_run_claimed_the_slot = eval_routes._claim_run_slot(_app_state())
    assert other_run_claimed_the_slot, "the slot starts free"

    resp = client.post("/eval/runs")

    assert resp.status_code == 409
    assert resp.json() == {"detail": "An eval run is already in progress"}
    assert runner.started_with == []


def test_a_finished_run_frees_the_slot_for_the_next_start(client, runner, eval_dir):
    """The slot is released when a run completes, so runs can follow one another."""
    write_eval_set(eval_dir, "golden.yml", "golden-qa-v1")

    assert client.post("/eval/runs").status_code == 202
    assert client.post("/eval/runs").status_code == 202
    assert len(runner.executed) == 2


def test_a_run_that_fails_in_the_background_is_logged_and_frees_the_slot(
    client, runner, eval_dir, caplog
):
    """A failure after the reply cannot change the reply, is logged, and blocks nothing later.

    By then the runner has already marked the run failed in the database.
    """
    write_eval_set(eval_dir, "golden.yml", "golden-qa-v1")
    runner.execute_error = EvalRunnerError("http://localhost:8000/ask returned HTTP 503")

    failed = client.post("/eval/runs")
    runner.execute_error = None
    next_start = client.post("/eval/runs")

    assert failed.status_code == 202
    assert f"Eval run {RUN_ID} failed" in caplog.text
    assert next_start.status_code == 202


def test_post_builds_a_runner_that_calls_this_server_and_judges_with_its_llm(
    client, eval_dir, monkeypatch
):
    """The runner calls the app over loopback with the admin token; the judge uses the LLM settings.

    The judge client gets the long judge timeout, not the server's shorter request timeout.
    """
    from api.main import app

    settings = Settings(
        app_port=8321,
        admin_token=SecretStr("admin-secret"),
        llm_base_url="http://llm.test/v1",
        llm_api_key=SecretStr("llm-secret"),
        llm_model="judge-model",
    )
    monkeypatch.setattr(app.state, "settings", settings)
    judge = object()
    judge_args: dict[str, Any] = {}
    runner_args: dict[str, Any] = {}

    def make_judge(**kwargs: Any) -> object:
        """Record how the endpoint builds the judge client."""
        judge_args.update(kwargs)
        return judge

    def make_runner(**kwargs: Any) -> FakeRunner:
        """Record how the endpoint builds the runner."""
        runner_args.update(kwargs)
        return FakeRunner()

    monkeypatch.setattr(eval_routes, "OpenAICompatibleClient", make_judge)
    monkeypatch.setattr(eval_routes, "EvalRunner", make_runner)
    write_eval_set(eval_dir, "golden.yml", "golden-qa-v1")

    assert client.post("/eval/runs").status_code == 202

    assert judge_args == {
        "base_url": "http://llm.test/v1",
        "api_key": "llm-secret",
        "model": "judge-model",
        "timeout": JUDGE_TIMEOUT_SECONDS,
        "reasoning_effort": "none",
    }
    assert runner_args == {
        "base_url": "http://localhost:8321",
        "llm_client": judge,
        "session_factory": SESSION_FACTORY,
        "admin_token": "admin-secret",
    }


# --- Database and auth ---


@pytest.mark.parametrize(("method", "path"), _ROUTES)
def test_returns_503_when_the_database_is_unavailable(client_no_db, method, path):
    """Without a database session factory every eval endpoint answers 503."""
    resp = client_no_db.request(method, path)

    assert resp.status_code == 503
    assert resp.json() == {"detail": "Database unavailable"}


@pytest.mark.parametrize(("method", "path"), _ROUTES)
def test_every_route_requires_the_admin_token(auth_client, method, path):
    """The router carries the admin check, so no eval endpoint answers without a token."""
    resp = auth_client.request(method, path)

    assert resp.status_code == 401
    assert resp.json() == {"detail": "Missing Bearer token"}
