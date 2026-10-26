"""Admin endpoints for the evaluation pipeline: read results and start runs.

The read endpoints query the eval tables and the v_eval_run_summary view.
POST /eval/runs starts the same EvalRunner the CLI uses, inside this process.
It answers as soon as the run exists and scores the questions in a background task, because a run
takes many minutes.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.datastructures import State

from api.dependencies import DbSession, require_admin_token, require_db_session
from geoai_rag.config import Settings, get_settings
from geoai_rag.db.models import EvalQuestion, EvalResult, EvalRun, EvalSet, VEvalRunSummary
from geoai_rag.eval.runner import (
    JUDGE_TIMEOUT_SECONDS,
    EvalRunner,
    EvalRunnerError,
    NoMatchingQuestionsError,
    StartedRun,
    load_eval_set,
)
from geoai_rag.generation.llm_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)


def _eval_set_dir() -> Path:
    """Return the eval set directory from settings. Testable via monkeypatch."""
    return Path(get_settings().eval_data_dir)


# The largest value of a Postgres integer column. A bigger id in the URL could only fail inside
# the database, as a 500, so it is rejected as invalid instead.
_MAX_ID = 2**31 - 1
_Id = Annotated[int, PathParam(ge=1, le=_MAX_ID)]

# The app.state attribute that is True while a run started by POST /eval/runs is in progress.
_RUN_ACTIVE = "eval_run_active"

_INVALID_EVAL_SET = "Eval set file is invalid; see the server log"

router = APIRouter(tags=["eval"], dependencies=[Depends(require_admin_token)])


class EvalRunSummary(BaseModel):
    """Aggregate scores of one eval run, read from the v_eval_run_summary view.

    Each average covers the questions stored so far and is None while there are none.
    """

    model_config = ConfigDict(from_attributes=True)

    eval_run_id: int
    eval_set_id: int
    deployment_id: int
    status: str
    source: str
    created_at: datetime
    planned_count: int | None
    last_heartbeat_at: datetime | None
    deployment_label: str | None
    questions_evaluated: int
    avg_retrieval_precision: float | None
    avg_retrieval_recall: float | None
    avg_answer_similarity: float | None
    avg_faithfulness: float | None


class QuestionResult(BaseModel):
    """The answer the pipeline gave to one question in a run, and how it scored.

    question and expected_answer are the texts stored with the result, so they show what was
    asked in that run even if the question was reworded or deleted afterwards.
    question_key is the question's id in the eval set file, or None when its row is gone.
    scores maps each metric name, such as retrieval_precision or faithfulness, to its value.
    trace holds the timings, the retrieved chunks, and the judge's replies.
    """

    question: str
    question_key: str | None
    expected_answer: str | None
    response: str
    scores: dict[str, float]
    trace: dict[str, Any]


class EvalRunDetail(BaseModel):
    """One eval run with the result of every question stored so far.

    status is pending, running, completed, or failed.
    config is the pipeline configuration the run started with, or None if none was recorded.
    """

    eval_run_id: int
    eval_set_id: int
    status: str
    config: dict[str, Any] | None
    created_at: datetime
    results: list[QuestionResult]


class EvalSetSummary(BaseModel):
    """An eval set and the number of questions it holds."""

    id: int
    name: str
    description: str | None
    question_count: int
    created_at: datetime


class SetQuestion(BaseModel):
    """One question of an eval set, with the answer and sources its result is judged against.

    question_key is None for a question saved before keys existed.
    tags is an empty list for a question saved without tags.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    question_key: str | None
    question: str
    expected_answer: str | None
    expected_sources: list[str]
    tags: list[str]

    @field_validator("tags", mode="before")
    @classmethod
    def null_tags_become_empty(cls, value: Any) -> Any:
        """Read a NULL tags column as an empty list."""
        return [] if value is None else value


class StartRunRequest(BaseModel):
    """Which eval set to run and which of its questions; every field is optional.

    eval_set_name is the name field of an eval set file.
    When it is omitted, the first eval set file in file name order is run.
    tags and question_ids narrow the run to some of the questions, as the CLI flags do.
    Every id in question_ids must exist in the eval set, or no run starts.
    To run every question, leave question_ids out; an empty list is rejected.
    Unknown fields are rejected so that a misspelled name cannot silently run the default set.
    resume_run_id resumes a failed or cancelled run instead of starting a new one.
    When set, eval_set_name, tags, and question_ids are ignored.
    """

    model_config = ConfigDict(extra="forbid")

    eval_set_name: str | None = Field(default=None, min_length=1, max_length=200)
    tags: list[Annotated[str, Field(min_length=1, max_length=100)]] | None = Field(
        default=None, max_length=50
    )
    question_ids: list[Annotated[str, Field(min_length=1, max_length=100)]] | None = Field(
        default=None, min_length=1
    )
    resume_run_id: int | None = Field(default=None, ge=1, le=_MAX_ID)


class StartRunResponse(BaseModel):
    """The run that was started; poll GET /eval/runs/{eval_run_id} for its progress."""

    eval_run_id: int
    status: Literal["running", "completed"]


@router.get("/runs", response_model=list[EvalRunSummary])
async def list_runs(
    session: DbSession,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
):
    """Return eval runs with their average scores, newest first."""
    result = await session.execute(
        select(VEvalRunSummary)
        .order_by(VEvalRunSummary.created_at.desc(), VEvalRunSummary.eval_run_id.desc())
        .limit(limit)
    )
    return [EvalRunSummary.model_validate(row) for row in result.scalars().all()]


@router.get(
    "/runs/{run_id}",
    response_model=EvalRunDetail,
    responses={404: {"description": "No such eval run"}},
)
async def get_run(
    run_id: _Id,
    session: DbSession,
):
    """Return one run with the result of every question stored so far.

    Results are stored one question at a time, so a run that is still going has only some.
    """
    run = await session.get(EvalRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    result = await session.execute(
        select(EvalResult, EvalQuestion.question_key)
        .select_from(EvalResult)
        .outerjoin(EvalQuestion, EvalResult.question_id == EvalQuestion.id)
        .where(EvalResult.eval_run_id == run_id)
        .order_by(EvalResult.id)
    )
    return EvalRunDetail(
        eval_run_id=run.id,
        eval_set_id=run.eval_set_id,
        status=run.status,
        config=run.config,
        created_at=run.created_at,
        results=[
            QuestionResult(
                question=stored.question_snapshot,
                question_key=question_key,
                expected_answer=stored.expected_answer_snapshot,
                response=stored.response,
                scores=stored.scores,
                trace=stored.trace,
            )
            for stored, question_key in result.all()
        ],
    )


@router.get(
    "/runs/{run_id}/export",
    dependencies=[Depends(require_admin_token), Depends(require_db_session)],
    responses={404: {"description": "No such eval run or no results"}},
)
async def export_run(run_id: _Id, session: DbSession):
    """Download all results of one eval run as a CSV file."""
    result = await session.execute(
        select(EvalResult, EvalQuestion.question_key)
        .select_from(EvalResult)
        .outerjoin(EvalQuestion, EvalResult.question_id == EvalQuestion.id)
        .where(EvalResult.eval_run_id == run_id)
        .order_by(EvalResult.id)
    )
    rows = result.all()
    if not rows:
        raise HTTPException(status_code=404, detail="No results found for this eval run")

    fieldnames = [
        "question_key",
        "question",
        "expected_answer",
        "response",
        "retrieval_precision",
        "retrieval_recall",
        "answer_similarity",
        "faithfulness",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for stored, question_key in rows:
        scores = stored.scores or {}
        writer.writerow(
            {
                "question_key": question_key,
                "question": stored.question_snapshot,
                "expected_answer": stored.expected_answer_snapshot,
                "response": stored.response,
                "retrieval_precision": scores.get("retrieval_precision"),
                "retrieval_recall": scores.get("retrieval_recall"),
                "answer_similarity": scores.get("answer_similarity"),
                "faithfulness": scores.get("faithfulness"),
            }
        )

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="eval_run_{run_id}.csv"'},
    )


@router.get("/sets", response_model=list[EvalSetSummary])
async def list_sets(session: DbSession):
    """Return every eval set with its question count, oldest first."""
    result = await session.execute(
        select(EvalSet, func.count(EvalQuestion.id))
        .select_from(EvalSet)
        .outerjoin(EvalQuestion, EvalQuestion.eval_set_id == EvalSet.id)
        .group_by(EvalSet.id)
        .order_by(EvalSet.id)
    )
    return [
        EvalSetSummary(
            id=eval_set.id,
            name=eval_set.name,
            description=eval_set.description,
            question_count=question_count,
            created_at=eval_set.created_at,
        )
        for eval_set, question_count in result.all()
    ]


@router.get(
    "/sets/{set_id}/questions",
    response_model=list[SetQuestion],
    responses={404: {"description": "No such eval set"}},
)
async def list_set_questions(
    set_id: _Id,
    session: DbSession,
):
    """Return the questions of one eval set in the order they were added."""
    if await session.get(EvalSet, set_id) is None:
        raise HTTPException(status_code=404, detail="Eval set not found")
    result = await session.execute(
        select(EvalQuestion).where(EvalQuestion.eval_set_id == set_id).order_by(EvalQuestion.id)
    )
    return [SetQuestion.model_validate(row) for row in result.scalars().all()]


@router.post(
    "/runs",
    status_code=202,
    response_model=StartRunResponse,
    # Declared for its 503 handling, not for its session: it rejects the request when there is no
    # database and turns a lost connection raised below into a 503.
    # The run opens its own sessions from the session factory, because it outlives the request.
    dependencies=[Depends(require_db_session)],
    responses={
        404: {"description": "No such eval set"},
        409: {"description": "An eval run is already in progress"},
        422: {"description": "Invalid request, an unknown question id, or no question selected"},
        503: {"description": "Database unavailable, or the run could not start"},
    },
)
async def start_eval_run(
    request: Request,
    background_tasks: BackgroundTasks,
    body: StartRunRequest | None = None,
):
    """Start an eval run and reply at once; its questions are scored in a background task.

    The reply is 202 with the id of the new run.
    Results are stored one question at a time, so GET /eval/runs/{id} shows the progress and the
    status becomes completed or failed at the end.
    Only one run started here can be active at a time.
    The reason a run failed is only in the server log, because eval_runs has no column for it.
    """
    options = body if body is not None else StartRunRequest()
    state = request.app.state
    if not _claim_run_slot(state):
        raise HTTPException(status_code=409, detail="An eval run is already in progress")
    try:
        if options.resume_run_id is not None:
            runner, started = await _resume_or_raise(
                state.settings, state.db_session_factory, options.resume_run_id
            )
            if not started.questions:
                _release_run_slot(state)
                logger.info(
                    "Resume run %d: all questions already scored, marked completed",
                    started.run_id,
                )
                return StartRunResponse(eval_run_id=started.run_id, status="completed")
        else:
            runner, started = await _start_or_raise(
                state.settings, state.db_session_factory, options
            )
    except BaseException:
        _release_run_slot(state)
        raise
    logger.info(
        "Started eval run %d: %s, %d of %d question(s)",
        started.run_id,
        started.eval_set_name,
        len(started.questions),
        started.set_size,
    )
    background_tasks.add_task(_execute_in_background, state, runner, started)
    return StartRunResponse(eval_run_id=started.run_id, status="running")


def _claim_run_slot(state: State) -> bool:
    """Mark that a run is active and return True, or return False when one already is.

    The check and the mark have no await between them, so two requests cannot both claim the
    slot.
    The mark lives in this process only, which is enough because the app runs as one process.
    """
    if getattr(state, _RUN_ACTIVE, False):
        return False
    setattr(state, _RUN_ACTIVE, True)
    return True


def _release_run_slot(state: State) -> None:
    """Mark that no run is active."""
    setattr(state, _RUN_ACTIVE, False)


def _find_eval_set_file(name: str | None) -> Path | None:
    """Return the eval set file to run, or None when there is none.

    With no name, the first file in file name order is used, the order scripts/run_eval.py uses.
    With a name, files are read in that order and the first whose name field matches is used.
    The name field is matched, not the file name, because the field is what the database stores.

    Raises:
        ValueError: If a file read on the way is not a valid eval set.
    """
    eval_dir = _eval_set_dir()
    files = sorted([*eval_dir.glob("*.yml"), *eval_dir.glob("*.yaml")])
    if name is None:
        return files[0] if files else None
    return next((path for path in files if load_eval_set(path)["name"] == name), None)


def _missing_eval_set_detail(name: str | None) -> str:
    """Describe why no eval set file was found for a start request."""
    return "No eval set files found" if name is None else f"Eval set '{name}' not found"


def _build_runner(
    settings: Settings, session_factory: async_sessionmaker[AsyncSession]
) -> EvalRunner:
    """Build a runner that evaluates this server through its own HTTP API.

    The runner calls the app over loopback on its own port, as scripts/run_eval.py does by
    default, so no traffic leaves the machine.
    The judge LLM is built from the server's LLM settings but with the longer judge timeout,
    because a reasoning model can think for minutes and the server's request timeout is shorter.
    """
    judge = OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        timeout=JUDGE_TIMEOUT_SECONDS,
        reasoning_effort=settings.eval_judge_reasoning_effort,
    )
    return EvalRunner(
        base_url=settings.local_api_url,
        llm_client=judge,
        session_factory=session_factory,
        admin_token=settings.admin_token.get_secret_value(),
    )


async def _start_or_raise(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    options: StartRunRequest,
) -> tuple[EvalRunner, StartedRun]:
    """Find the eval set and create its run row, turning each expected failure into an HTTP error.

    Raises:
        HTTPException: 404 when no eval set file matches, 422 when the filters select no
            question or name a question id the eval set lacks, 503 when the runner's own
            preconditions fail, and 500 when an eval set file is invalid.
    """
    try:
        eval_set_file = await asyncio.to_thread(_find_eval_set_file, options.eval_set_name)
        if eval_set_file is None:
            detail = _missing_eval_set_detail(options.eval_set_name)
            raise HTTPException(status_code=404, detail=detail)
        runner = _build_runner(settings, session_factory)
        started = await runner.start_run(
            eval_set_file, tags=options.tags, question_ids=options.question_ids, source="api"
        )
    except NoMatchingQuestionsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        # The message names a path on the server, so it goes to the log and not to the caller.
        logger.exception("Eval set file is invalid")
        raise HTTPException(status_code=500, detail=_INVALID_EVAL_SET) from exc
    except EvalRunnerError as exc:
        logger.warning("Eval run could not start: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return runner, started


async def _resume_or_raise(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    run_id: int,
) -> tuple[EvalRunner, StartedRun]:
    """Resume a failed or cancelled eval run, turning expected failures into HTTP errors.

    Raises:
        HTTPException: 404 when the run does not exist, 409 when its status does not
            allow resuming, and 503 when the runner's own preconditions fail.
    """
    try:
        runner = _build_runner(settings, session_factory)
        started = await runner.resume_run(run_id)
    except EvalRunnerError as exc:
        detail = str(exc)
        if "not found" in detail.lower():
            raise HTTPException(status_code=404, detail=detail) from exc
        raise HTTPException(status_code=409, detail=detail) from exc
    return runner, started


async def _execute_in_background(state: State, runner: EvalRunner, started: StartedRun) -> None:
    """Score a started run's questions, then free the run slot. Never raises.

    The runner marks the run failed when it stops early, so a failure here only needs logging.
    An exception that escaped would surface after the reply was sent, where nothing handles it.
    """
    try:
        await runner.execute_run(started)
        logger.info("Eval run %d completed", started.run_id)
    except Exception:
        logger.exception("Eval run %d failed", started.run_id)
    finally:
        _release_run_slot(state)


@router.post(
    "/runs/{run_id}/cancel",
    dependencies=[Depends(require_admin_token)],
    status_code=200,
    responses={
        404: {"description": "No such eval run"},
        409: {"description": "Run is not active"},
    },
)
async def cancel_run(run_id: _Id, session: DbSession):
    """Cancel a running or pending eval run.

    Sets the run status to cancelled.
    The background task checks the status between questions and will stop.
    """
    run = await session.get(EvalRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    if run.status not in ("running", "pending"):
        raise HTTPException(status_code=409, detail="Run is not active")
    run.status = "cancelled"
    await session.commit()
    return {"status": "cancelled", "run_id": run_id}
