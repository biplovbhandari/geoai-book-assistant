"""Orchestrates evaluation runs against a running book assistant API.

For each golden question the runner calls POST /ask, scores retrieval from the source paths of the
retrieved chunks, asks a judge LLM to rate answer similarity and faithfulness, and stores one
eval_results row.
The scoring maths and the judge prompts live in geoai_rag.eval.scorer; this module owns the
I/O: the YAML file, the HTTP calls to the API, the judge LLM calls, and the database writes.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Self

import requests
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from geoai_rag.db.models import Deployment, EvalQuestion, EvalResult, EvalRun, EvalSet
from geoai_rag.eval.scorer import (
    build_faithfulness_prompt,
    build_similarity_prompt,
    parse_judge_score,
    retrieval_precision,
    retrieval_recall,
)
from geoai_rag.interfaces import LLMClient
from geoai_rag.models import Answer

logger = logging.getLogger(__name__)

# POST /ask retrieves, reranks, and generates with a local LLM, which takes about a minute.
ASK_TIMEOUT_SECONDS = 300.0
CONFIG_TIMEOUT_SECONDS = 30.0
# A judge reply is one number, but a reasoning model thinks first, so leave it room to think.
JUDGE_MAX_TOKENS = 4096
# Judging must be repeatable from run to run, so sampling noise is switched off.
JUDGE_TEMPERATURE = 0.0
# How long one judge call may take, for whoever builds the judge client (the CLI and the API).
# A reasoning model can think for minutes, so this is far above the server's own LLM timeout.
JUDGE_TIMEOUT_SECONDS = 300.0
QUESTION_MAX_RETRIES = 2
QUESTION_RETRY_BASE_DELAY = 5

_CHUNK_SEPARATOR = "\n\n---\n\n"


class EvalRunnerError(RuntimeError):
    """A run cannot continue: the API, the judge LLM, or a database precondition failed."""


class NoMatchingQuestionsError(ValueError):
    """The question filters cannot be met by an eval set.

    Either no question passes them, or a requested question id is not in the eval set.
    """


_NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _GoldenQuestion(BaseModel):
    """One question entry in an eval set YAML file."""

    model_config = ConfigDict(extra="forbid")

    id: _NonEmptyStr
    question: _NonEmptyStr
    expected_answer: _NonEmptyStr
    expected_sources: list[_NonEmptyStr] = Field(min_length=1)
    tags: list[_NonEmptyStr] = Field(default_factory=list)
    notes: str | None = None


class _EvalSetFile(BaseModel):
    """The top level of an eval set YAML file."""

    model_config = ConfigDict(extra="forbid")

    name: _NonEmptyStr
    description: str | None = None
    questions: list[_GoldenQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def check_unique_ids_and_texts(self) -> Self:
        """Reject repeats: ids select questions and key their rows in the database.

        Texts must also be unique, because rows saved without a question_key are matched by text.
        """
        for label, values in (
            ("id", [q.id for q in self.questions]),
            ("text", [q.question for q in self.questions]),
        ):
            repeated = sorted({value for value in values if values.count(value) > 1})
            if repeated:
                raise ValueError(f"duplicate question {label}: {', '.join(repeated)}")
        return self


@dataclass(frozen=True)
class _Scored:
    """What scoring one question produced, before it is stored."""

    answer: dict[str, Any]
    trace: dict[str, Any]
    scores: dict[str, float]


@dataclass(frozen=True)
class StartedRun:
    """A run whose eval_runs row exists with status 'running' but whose questions are unasked.

    A started run must be passed to EvalRunner.execute_run, or its row stays 'running' for good.

    run_id is the eval_runs row id.
    eval_set_name is the name of the eval set.
    set_size is the number of questions in the whole eval set, which can exceed len(questions).
    questions are the questions to ask in file order, each with its eval_questions id as db_id.
    """

    run_id: int
    eval_set_name: str
    set_size: int
    questions: list[dict[str, Any]]


async def sync_eval_set_to_db(
    session_factory: async_sessionmaker[AsyncSession],
    eval_set_data: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    """Upsert an eval set and its questions into the database.

    Standalone version of EvalRunner.sync_to_db for use at app startup
    when no runner is available.

    The eval set is found by name.
    A question is matched to its row by its YAML id, which is stored as question_key.
    Rewording a question therefore updates its row and keeps the link to earlier results.
    A row saved before question_key existed has no key, so it is matched by question text
    instead and gets its key then.
    Rows whose question was removed from the YAML are left in place so earlier results keep
    their link.

    Args:
        session_factory: Async SQLAlchemy session factory for database writes.
        eval_set_data: An eval set as returned by load_eval_set.

    Returns:
        The eval set's database id and every YAML question with its ``db_id`` added.
    """
    name = eval_set_data["name"]
    questions = eval_set_data["questions"]
    async with session_factory() as session:
        result = await session.execute(select(EvalSet).where(EvalSet.name == name))
        eval_set = result.scalar_one_or_none()
        if eval_set is None:
            eval_set = EvalSet(name=name, description=eval_set_data["description"])
            session.add(eval_set)
            await session.flush()
        else:
            eval_set.description = eval_set_data["description"]

        result = await session.execute(
            select(EvalQuestion)
            .where(EvalQuestion.eval_set_id == eval_set.id)
            .order_by(EvalQuestion.id)
        )
        by_key: dict[str, EvalQuestion] = {}
        unkeyed_by_text: dict[str, EvalQuestion] = {}
        for saved in result.scalars().all():
            if saved.question_key is None:
                # Ordered by id, so if a text was saved twice the oldest row is the one reused.
                unkeyed_by_text.setdefault(saved.question, saved)
            else:
                by_key[saved.question_key] = saved
        rows = []
        for question in questions:
            columns = _question_columns(question)
            row = by_key.get(question["id"])
            if row is None:
                row = unkeyed_by_text.pop(question["question"], None)
            if row is None:
                row = EvalQuestion(eval_set_id=eval_set.id, **columns)
                session.add(row)
            else:
                for column, value in columns.items():
                    setattr(row, column, value)
            rows.append(row)
        await session.flush()
        # Read the ids before committing so this works for any session factory, not only
        # one built with expire_on_commit=False.
        eval_set_id = eval_set.id
        synced = [{**q, "db_id": row.id} for q, row in zip(questions, rows, strict=True)]
        await session.commit()
    return eval_set_id, synced


def load_eval_set(yaml_path: Path) -> dict[str, Any]:
    """Load and validate an eval set YAML file.

    Args:
        yaml_path: Path of the YAML file.

    Returns:
        A dict with name, description, and questions.
        Every question has id, question, expected_answer, expected_sources, tags, and notes,
        with surrounding whitespace stripped and tags defaulting to an empty list.

    Raises:
        FileNotFoundError: If yaml_path is not a file.
        ValueError: If the file is not valid YAML or breaks the eval set format.
    """
    # pyyaml is installed in the indexer and dev groups only, not in the API image, so it is
    # imported here to keep importing this module safe there.
    import yaml

    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Eval set file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    try:
        return _EvalSetFile.model_validate(raw).model_dump()
    except ValidationError as exc:
        raise ValueError(f"{path} is not a valid eval set:\n{exc}") from exc


def select_questions(
    eval_set_data: dict[str, Any],
    tags: list[str] | None = None,
    question_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Pick the questions to run, in file order.

    A tag filter keeps the questions that carry at least one of the given tags.
    An id filter keeps the questions whose YAML id is in the given list.
    When both filters are given, a question must pass both.
    Every id in the id filter must exist in the eval set, whatever the tag filter does with it.
    A mistyped id therefore fails the selection instead of quietly running fewer questions.

    Args:
        eval_set_data: An eval set as returned by load_eval_set.
        tags: Keep only questions with at least one of these tags. None or empty keeps all.
        question_ids: Keep only the questions with these YAML ids. None keeps all.
            An empty list is not the same as None: it selects nothing, so it raises.

    Raises:
        NoMatchingQuestionsError: If an id in question_ids is not in the eval set, or if no
            question passes the filters.
    """
    name = eval_set_data["name"]
    selected = list(eval_set_data["questions"])
    if question_ids is not None:
        wanted_ids = set(question_ids)
        unknown_ids = sorted(wanted_ids - {q["id"] for q in selected})
        if unknown_ids:
            listed = ", ".join(map(repr, unknown_ids))
            raise NoMatchingQuestionsError(f"Question id(s) not found in '{name}': {listed}")
        selected = [q for q in selected if q["id"] in wanted_ids]
    if tags:
        wanted_tags = set(tags)
        selected = [q for q in selected if wanted_tags.intersection(q["tags"])]
    if not selected:
        filters = []
        if question_ids is not None:
            filters.append(f"question ids {', '.join(map(repr, question_ids)) or '(none)'}")
        if tags:
            filters.append(f"tags {', '.join(tags)}")
        raise NoMatchingQuestionsError(f"No question in '{name}' matches {' and '.join(filters)}")
    return selected


def preview_eval_set(
    yaml_path: Path,
    tags: list[str] | None = None,
    question_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Print the questions a run would evaluate, without any API, judge, or database call.

    Returns:
        {"eval_set": name, "dry_run": True, "question_ids": ids of the selected questions}.

    Raises:
        FileNotFoundError: If yaml_path does not exist.
        ValueError: If the file breaks the eval set format.
        NoMatchingQuestionsError: If the filters select no question, or an id in question_ids
            is not in the eval set.
    """
    eval_set_data = load_eval_set(yaml_path)
    selected = select_questions(eval_set_data, tags, question_ids)
    total = len(eval_set_data["questions"])
    print(f"Dry run: {eval_set_data['name']}, {len(selected)} of {total} questions would run")
    for question in selected:
        print(f"  {question['id']}  {_truncate(question['question'], 70)}  {question['tags']}")
    return {
        "eval_set": eval_set_data["name"],
        "dry_run": True,
        "question_ids": [question["id"] for question in selected],
    }


def fetch_pipeline_config(base_url: str, admin_token: str = "") -> dict[str, Any]:
    """Fetch the server's active pipeline configuration from GET /config.

    Args:
        base_url: API base URL, for example http://localhost:8000.
        admin_token: The server's ADMIN_TOKEN, which that endpoint requires.

    Raises:
        EvalRunnerError: If the server cannot be reached, refuses the token, or replies badly.
    """
    url = f"{base_url.rstrip('/')}/config"
    body = _send(requests.get, url, headers=_bearer(admin_token), timeout=CONFIG_TIMEOUT_SECONDS)
    if not isinstance(body, dict):
        raise EvalRunnerError(f"{url} did not return a JSON object")
    return body


def format_scores(scores: dict[str, float]) -> str:
    """Render scores on one line, each labelled with the last word of its metric name."""
    return "  ".join(f"{_metric_label(name)} {value:.2f}" for name, value in scores.items())


def _metric_label(name: str) -> str:
    """Shorten a metric name to its last word, for example retrieval_recall to recall."""
    return name.rsplit("_", 1)[-1]


def _truncate(text: str, width: int) -> str:
    """Shorten text to at most width characters, ending with ... when it was cut."""
    return text if len(text) <= width else f"{text[: width - 3]}..."


def _bearer(token: str) -> dict[str, str]:
    """Return an Authorization header for token, or no headers when the token is empty."""
    return {"Authorization": f"Bearer {token}"} if token else {}


def _send(send: Callable[..., requests.Response], url: str, **kwargs: Any) -> Any:
    """Make one HTTP request and return its decoded JSON body.

    Raises:
        EvalRunnerError: If the server cannot be reached, answers with an error status, or
            returns a body that is not JSON.
    """
    try:
        response = send(url, **kwargs)
    except requests.RequestException as exc:
        raise EvalRunnerError(f"Request to {url} failed: {exc}") from exc
    if not response.ok:
        detail = _error_detail(response)
        raise EvalRunnerError(f"{url} returned HTTP {response.status_code}: {detail}")
    try:
        return response.json()
    except ValueError as exc:
        raise EvalRunnerError(f"{url} returned a body that is not JSON") from exc


def _error_detail(response: requests.Response) -> str:
    """Return the server's error message, preferring the FastAPI ``detail`` field."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    detail = body.get("detail", body) if isinstance(body, dict) else body
    return str(detail)[:200]


def _source_path(source: dict[str, Any]) -> str:
    """Return a retrieved chunk's source path, or an empty string when its metadata has none."""
    return source["chunk"]["metadata"].get("source_path") or ""


def _format_chunks(sources: list[dict[str, Any]]) -> str:
    """Render retrieved chunks as numbered blocks for the faithfulness judge.

    The numbering and layout follow the context the generator saw, so a marker such as [2] in
    the answer points at the same chunk in the judge's prompt.
    """
    blocks = [
        f"[{number}] {_source_path(source) or 'unknown'}\n{source['chunk']['content']}"
        for number, source in enumerate(sources, 1)
    ]
    return _CHUNK_SEPARATOR.join(blocks)


def _judge_score(reply: str, question_id: str, metric: str) -> float:
    """Parse a judge reply into a score, logging every 0.0 because it may be only a default.

    parse_judge_score returns 0.0 both for a genuine lowest rating and for a reply that holds
    no number, so each 0.0 is logged together with the reply that produced it.
    """
    score = parse_judge_score(reply)
    if score == 0.0:
        logger.warning(
            "Question %s: %s judged 0.0 from reply %r (a reply without a number also gives 0.0)",
            question_id,
            metric,
            reply[:200],
        )
    return score


def _build_trace(answer: dict[str, Any], judge_replies: dict[str, str]) -> dict[str, Any]:
    """Collect timings, retrieved chunks, and the raw judge replies into the result's trace."""
    return {
        "round_trip_ms": answer["round_trip_ms"],
        "retrieval_ms": answer["retrieval_ms"],
        "generation_ms": answer["generation_ms"],
        "chunks": [
            {
                "chunk_id": source["chunk"]["id"],
                "source_path": _source_path(source),
                "score": source["score"],
                "ranking_method": source["ranking_method"],
            }
            for source in answer["sources"]
        ],
        "judge_replies": judge_replies,
    }


def _question_columns(question: dict[str, Any]) -> dict[str, Any]:
    """Pick the eval_questions columns that are filled from a YAML question."""
    return {
        "question_key": question["id"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "expected_sources": question["expected_sources"],
        "tags": question["tags"],
    }


def _summarize(run_id: int, eval_set_name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the summary a finished run returns, with each metric averaged over its questions."""
    return {
        "eval_run_id": run_id,
        "eval_set": eval_set_name,
        "status": "completed",
        "question_count": len(results),
        "averages": {
            metric: statistics.fmean(result["scores"][metric] for result in results)
            for metric in results[0]["scores"]
        },
        "results": results,
    }


def _print_progress(position: int, total: int, result: dict[str, Any]) -> None:
    """Print one line for a question that was just scored and stored."""
    width = len(str(total))
    seconds = result["round_trip_ms"] / 1000
    print(
        f"[{position:>{width}}/{total}] {result['id']}  {format_scores(result['scores'])}"
        f"  (ask {seconds:.1f}s)",
        flush=True,
    )


def _table_row(first: str, second: str, cells: list[str]) -> str:
    """Lay out one table row: a short id, a question column, then right-aligned score cells."""
    return f"{first:<6}{second:<42}" + "".join(f"{cell:>13}" for cell in cells)


def _print_summary(summary: dict[str, Any]) -> None:
    """Print a finished run as a table of per-question scores followed by the means."""
    header = _table_row("ID", "Question", [_metric_label(name) for name in summary["averages"]])
    rule = "-" * len(header)
    print(f"\nEval run {summary['eval_run_id']} ({summary['eval_set']}): {summary['status']}")
    print(header)
    print(rule)
    for result in summary["results"]:
        cells = [f"{score:.2f}" for score in result["scores"].values()]
        print(_table_row(result["id"], _truncate(result["question"], 40), cells))
    print(rule)
    print(_table_row("Mean", "", [f"{score:.2f}" for score in summary["averages"].values()]))


class EvalRunner:
    """Runs eval sets against a live API and records the scored results in the database."""

    def __init__(
        self,
        base_url: str,
        llm_client: LLMClient,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        admin_token: str = "",
    ) -> None:
        """Initialize the eval runner.

        Args:
            base_url: API base URL, for example http://localhost:8000.
            llm_client: Client the judge calls go through, normally an OpenAICompatibleClient so
                that LLM_BASE_URL can point at Ollama, the NASA proxy, or any compatible endpoint.
                The client owns the model name, the API key, and the request timeout.
                A reasoning model can think for minutes, so give the client a long timeout.
            session_factory: Async SQLAlchemy session factory for all database writes.
            admin_token: The server's ADMIN_TOKEN, sent to GET /config.
        """
        self._base_url = base_url.rstrip("/")
        self._llm_client = llm_client
        self._session_factory = session_factory
        self._admin_token = admin_token

    @staticmethod
    def load_eval_set(yaml_path: Path) -> dict[str, Any]:
        """Load and validate an eval set YAML file; see the module-level load_eval_set."""
        return load_eval_set(yaml_path)

    async def sync_to_db(self, eval_set_data: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        """Make sure the eval set and all its questions exist in the database, up to date.

        The eval set is found by name.
        A question is matched to its row by its YAML id, which is stored as question_key.
        Rewording a question therefore updates its row and keeps the link to earlier results.
        A row saved before question_key existed has no key, so it is matched by question text
        instead and gets its key then.
        Rows whose question was removed from the YAML are left in place so earlier results keep
        their link.

        Args:
            eval_set_data: An eval set as returned by load_eval_set.

        Returns:
            The eval set's database id and every YAML question with its ``db_id`` added.
        """
        return await sync_eval_set_to_db(self._session_factory, eval_set_data)

    def get_config(self) -> dict[str, Any]:
        """Fetch the active pipeline configuration from GET /config."""
        return fetch_pipeline_config(self._base_url, self._admin_token)

    def ask_question(self, question: str) -> dict[str, Any]:
        """Ask the API one question through POST /ask and time the round trip.

        Args:
            question: The question text.

        Returns:
            The Answer fields (answer, sources, citations, retrieval_ms, generation_ms) plus
            round_trip_ms, the wall-clock milliseconds the request took.

        Raises:
            EvalRunnerError: If the request fails or the body is not an Answer.
        """
        url = f"{self._base_url}/ask"
        started = time.monotonic()
        body = _send(requests.post, url, json={"question": question}, timeout=ASK_TIMEOUT_SECONDS)
        round_trip_ms = int((time.monotonic() - started) * 1000)
        try:
            answer = Answer.model_validate(body)
        except ValidationError as exc:
            raise EvalRunnerError(f"POST {url} did not return an Answer: {exc}") from exc
        return {**answer.model_dump(mode="json"), "round_trip_ms": round_trip_ms}

    def judge_llm(self, messages: list[dict[str, str]]) -> str:
        """Send messages straight to the judge LLM and return its reply text.

        This bypasses POST /ask on purpose: a judge prompt must not go through retrieval or the
        book assistant's system prompt.

        Args:
            messages: Chat messages, each with a role and a content.

        Raises:
            EvalRunnerError: If the judge LLM call fails or the reply holds no text.
                The client's own error stays attached as the cause.
        """
        try:
            reply = self._llm_client.chat(
                messages, temperature=JUDGE_TEMPERATURE, max_tokens=JUDGE_MAX_TOKENS
            )
        except RuntimeError as exc:
            # OpenAICompatibleClient raises RuntimeError when the reply has no choices or no text.
            raise EvalRunnerError(
                f"Judge LLM returned no usable reply ({exc}); a reasoning model may have used "
                f"all {JUDGE_MAX_TOKENS} tokens thinking"
            ) from exc
        except Exception as exc:
            # The LLMClient interface names no exception type, so wrap whatever this client raises.
            raise EvalRunnerError(
                f"Judge LLM request failed ({type(exc).__name__}): {exc}"
            ) from exc
        return reply.strip()

    async def run(
        self,
        yaml_path: Path,
        tags: list[str] | None = None,
        question_ids: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Evaluate an eval set and store the scored results.

        The whole YAML file is synced to the database, then only the questions that pass the
        filters are asked.
        Each result is committed as soon as its question is scored, so a failure part-way keeps
        the earlier results.
        This is start_run followed by execute_run, which a caller can also call separately to
        learn the run id before the questions are asked.

        Args:
            yaml_path: Eval set YAML file.
            tags: Run only questions that carry at least one of these tags.
            question_ids: Run only the questions with these YAML ids. Each id must be in the
                eval set; see select_questions.
            dry_run: Print the questions that would run, then return without calling the API,
                the judge, or the database.

        Returns:
            For a dry run, {"eval_set", "dry_run", "question_ids"}.
            Otherwise {"eval_run_id", "eval_set", "status", "question_count", "averages",
            "results"}, where each result holds the question id, its text, its scores, and the
            /ask round trip in milliseconds.

        Raises:
            FileNotFoundError: If yaml_path does not exist.
            ValueError: If the file breaks the eval set format.
            NoMatchingQuestionsError: If the filters select no question, or an id in
                question_ids is not in the eval set.
            EvalRunnerError: If the API, the judge, or a database precondition fails.
                A run row that was already created is left with status 'failed'.
        """
        if dry_run:
            return preview_eval_set(yaml_path, tags, question_ids)

        started = await self.start_run(yaml_path, tags, question_ids)
        return await self.execute_run(started)

    async def start_run(
        self,
        yaml_path: Path,
        tags: list[str] | None = None,
        question_ids: list[str] | None = None,
        source: str = "cli",
    ) -> StartedRun:
        """Check a run request and create the run's eval_runs row, without asking any question.

        This is the part of run() that comes before the first question.
        It is public so that a caller can see every precondition failure, and learn the run id,
        before the slow part begins.
        The whole YAML file is synced to the database, then only the questions that pass the
        filters are selected.

        Args:
            yaml_path: Eval set YAML file.
            tags: Select only questions that carry at least one of these tags.
            question_ids: Select only the questions with these YAML ids. Each id must be in the
                eval set; see select_questions.

        Returns:
            The started run, whose row has status 'running'.
            It must be passed to execute_run, or the row stays 'running'.

        Raises:
            FileNotFoundError: If yaml_path does not exist.
            ValueError: If the file breaks the eval set format.
            NoMatchingQuestionsError: If the filters select no question, or an id in
                question_ids is not in the eval set.
            EvalRunnerError: If GET /config fails or no deployment is active.
        """
        eval_set_data = load_eval_set(yaml_path)
        selected = select_questions(eval_set_data, tags, question_ids)
        eval_set_id, synced = await self.sync_to_db(eval_set_data)
        db_ids = {question["id"]: question["db_id"] for question in synced}
        config = await asyncio.to_thread(self.get_config)
        run_id = await self._create_run_row(
            eval_set_id, config, source=source, planned_count=len(selected)
        )
        return StartedRun(
            run_id=run_id,
            eval_set_name=eval_set_data["name"],
            set_size=len(eval_set_data["questions"]),
            questions=[{**item, "db_id": db_ids[item["id"]]} for item in selected],
        )

    async def execute_run(self, started: StartedRun) -> dict[str, Any]:
        """Ask, score, and store each question of a started run, then mark the run completed.

        Each result is committed as soon as its question is scored, so a failure part-way keeps
        the earlier results.

        Args:
            started: A run returned by start_run.

        Returns:
            {"eval_run_id", "eval_set", "status", "question_count", "averages", "results"},
            where each result holds the question id, its text, its scores, and the /ask round
            trip in milliseconds.

        Raises:
            EvalRunnerError: If the API, the judge, or a database precondition fails.
                The run row is left with status 'failed'.
        """
        run_id = started.run_id
        name = started.eval_set_name
        total = started.set_size
        questions = started.questions
        print(f"Eval run {run_id}: {name}, {len(questions)} of {total} questions", flush=True)
        results: list[dict[str, Any]] = []
        failed_count = 0
        try:
            for position, question in enumerate(questions, 1):
                question_key = question["id"]
                for attempt in range(QUESTION_MAX_RETRIES + 1):
                    try:
                        scored = await self._score_question(question)
                        await self._store_result(run_id, question, scored)
                        results.append(
                            {
                                "id": question_key,
                                "question": question["question"],
                                "scores": scored.scores,
                                "round_trip_ms": scored.answer["round_trip_ms"],
                            }
                        )
                        _print_progress(position, len(questions), results[-1])
                        break  # success, move to next question
                    except (
                        EvalRunnerError,
                        RuntimeError,
                        TimeoutError,
                        ConnectionError,
                        OSError,
                    ) as exc:
                        if attempt < QUESTION_MAX_RETRIES:
                            delay = QUESTION_RETRY_BASE_DELAY * (2**attempt)
                            logger.warning(
                                "Question %s attempt %d/%d failed, retrying in %ds: %s",
                                question_key,
                                attempt + 1,
                                QUESTION_MAX_RETRIES + 1,
                                delay,
                                exc,
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.error(
                                "Question %s failed after %d attempts: %s",
                                question_key,
                                QUESTION_MAX_RETRIES + 1,
                                exc,
                            )
                            failed_count += 1
                            # Do NOT re-raise - continue to next question

            if failed_count == len(questions):
                await self._set_run_status(run_id, "failed")
            else:
                await self._set_run_status(run_id, "completed")
        except BaseException:
            await self._mark_failed(run_id)
            raise

        if failed_count:
            logger.warning(
                "Eval run %d: %d of %d question(s) failed",
                run_id,
                failed_count,
                len(questions),
            )

        if not results:
            summary: dict[str, Any] = {
                "eval_run_id": run_id,
                "eval_set": name,
                "status": "failed",
                "question_count": 0,
                "averages": {},
                "results": [],
            }
        else:
            summary = _summarize(run_id, name, results)
            if failed_count:
                summary["failed_count"] = failed_count
        _print_summary(summary)
        return summary

    async def _create_run_row(
        self,
        eval_set_id: int,
        config: dict[str, Any],
        source: str = "cli",
        planned_count: int | None = None,
    ) -> int:
        """Create the eval_runs row for the active deployment, with status 'running'.

        Args:
            eval_set_id: The eval set's database id.
            config: The pipeline configuration snapshot for this run.
            source: Where the run was triggered from ('cli' or 'api').
            planned_count: Number of questions this run will evaluate, if known.

        Raises:
            EvalRunnerError: If no deployment is active.
        """
        async with self._session_factory() as session:
            statement = select(Deployment).where(Deployment.is_active.is_(True))
            result = await session.execute(statement)
            try:
                deployment = result.scalar_one()
            except NoResultFound as exc:
                raise EvalRunnerError(
                    "No active deployment in the database. Start the API once against this "
                    "database so it records one, then run the evaluation again."
                ) from exc
            run = EvalRun(
                eval_set_id=eval_set_id,
                deployment_id=deployment.id,
                status="running",
                source=source,
                config=config,
                planned_count=planned_count,
            )
            session.add(run)
            await session.flush()
            run_id = run.id
            await session.commit()
        return run_id

    async def _score_question(self, question: dict[str, Any]) -> _Scored:
        """Ask the API one question, then score the answer: two retrieval and two judge metrics.

        The blocking HTTP calls run in worker threads, so the event loop stays free.
        That matters when the runner is hosted in the API process, whose own loop must be able
        to serve the POST /ask this method sends it.
        """
        answer = await asyncio.to_thread(self.ask_question, question["question"])
        sources = answer["sources"]
        retrieved_paths = [_source_path(source) for source in sources]
        similarity_reply = await asyncio.to_thread(
            self.judge_llm,
            build_similarity_prompt(
                question["question"], question["expected_answer"], answer["answer"]
            ),
        )
        faithfulness_reply = await asyncio.to_thread(
            self.judge_llm,
            build_faithfulness_prompt(answer["answer"], _format_chunks(sources)),
        )
        expected = question["expected_sources"]
        question_id = question["id"]
        scores = {
            "retrieval_precision": retrieval_precision(retrieved_paths, expected),
            "retrieval_recall": retrieval_recall(retrieved_paths, expected),
            "answer_similarity": _judge_score(similarity_reply, question_id, "answer_similarity"),
            "faithfulness": _judge_score(faithfulness_reply, question_id, "faithfulness"),
        }
        replies = {"answer_similarity": similarity_reply, "faithfulness": faithfulness_reply}
        return _Scored(answer=answer, trace=_build_trace(answer, replies), scores=scores)

    async def _store_result(self, run_id: int, question: dict[str, Any], scored: _Scored) -> None:
        """Write the eval_results row for one scored question."""
        async with self._session_factory() as session:
            session.add(
                EvalResult(
                    eval_run_id=run_id,
                    question_id=question["db_id"],
                    question_snapshot=question["question"],
                    expected_answer_snapshot=question["expected_answer"],
                    expected_sources_snapshot=question["expected_sources"],
                    response=scored.answer["answer"],
                    citations=scored.answer["citations"],
                    trace=scored.trace,
                    scores=scored.scores,
                )
            )
            await session.commit()

    async def _set_run_status(self, run_id: int, status: str) -> None:
        """Set the status of an eval run.

        Raises:
            EvalRunnerError: If the run row no longer exists.
        """
        async with self._session_factory() as session:
            run = await session.get(EvalRun, run_id)
            if run is None:
                raise EvalRunnerError(f"Eval run {run_id} no longer exists in the database")
            run.status = status
            await session.commit()

    async def _mark_failed(self, run_id: int) -> None:
        """Record the run as failed without hiding the error that stopped it."""
        try:
            await self._set_run_status(run_id, "failed")
        except Exception:
            logger.exception("Could not mark eval run %d as failed", run_id)
