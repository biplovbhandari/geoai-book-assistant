"""Admin debug endpoints for seeing what the RAG pipeline does with a question.

The dashboard's Retrieval Inspector and Pipeline Trace tabs use them:

- POST /debug/query runs retrieval only and returns the ranked chunks with each stage's scores.
- GET /debug/interactions and GET /debug/traces/{id} read the traces recorded for past chats.
- POST /debug/trace runs the whole pipeline for one question and returns its trace inline.
- POST /debug/groundedness asks the judge LLM which claims in a response the chunks support.

Nothing here writes to the database.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import func, select

from api.dependencies import (
    DbSession,
    acquire_llm_slot,
    require_admin_token,
    require_generation_ready,
    require_retrieval_ready,
)
from geoai_rag.config import TOP_K_MAX
from geoai_rag.db.models import (
    ChunkRecord,
    Interaction,
    InteractionCitation,
    InteractionTrace,
    PromptVersion,
)
from geoai_rag.eval.groundedness import build_groundedness_prompt, parse_groundedness_response
from geoai_rag.eval.judge_client import create_judge_client, judge_chat
from geoai_rag.eval.scorer import retrieval_ndcg, retrieval_precision
from geoai_rag.models.pipeline_result import PipelineResult

logger = logging.getLogger(__name__)

router = APIRouter(tags=["debug"], dependencies=[Depends(require_admin_token)])

# The stages the chat recorder writes to interaction_traces.
# Interactions recorded before trace enrichment have only retrieval and generation, so a
# "context" row is what marks an interaction as enriched.
_TRACE_STAGES = ("retrieval", "context", "generation")
_ENRICHED_STAGE = "context"


# --- Retrieval Inspector ---


class DebugQueryRequest(BaseModel):
    """A question to retrieve for, with optional expected source prefixes to score against."""

    question: str = Field(min_length=1, max_length=2000)
    expected_sources: list[str] | None = None


class ChunkResult(BaseModel):
    """One retrieved chunk.

    chunk_id is the id that each stage lists in its chunk_scores, so a stage's score can be
    matched to the chunk it scored.
    score is the final score, after the last stage.
    """

    rank: int
    chunk_id: str
    source_path: str
    text: str
    score: float


class DebugQueryResponse(BaseModel):
    """Retrieval settings in effect, the ranked chunks, stage scores, and optional metrics.

    precision and ndcg are set only when the request gave expected_sources.
    """

    question: str
    strategy: str
    top_k: int
    reranker: str
    rrf_k: int
    chunks: list[ChunkResult]
    stages: list[dict] = Field(default_factory=list)
    precision: float | None
    ndcg: float | None


@router.post(
    "/query", response_model=DebugQueryResponse, dependencies=[Depends(require_retrieval_ready)]
)
def debug_query(request: Request, body: DebugQueryRequest) -> DebugQueryResponse:
    """Run retrieval (no generation) and return the ranked chunks with each stage's scores."""
    settings = request.app.state.settings
    pipelines = request.app.state.pipelines

    try:
        result = pipelines.query.search(body.question, settings.retrieval_top_k)
    except Exception:
        logger.exception("Debug retrieval failed")
        raise HTTPException(status_code=503, detail="retrieval not ready")

    scored = result.chunks
    chunks = [
        ChunkResult(
            rank=rank,
            chunk_id=item.chunk.id,
            source_path=item.chunk.metadata.get("source_path") or "",
            text=item.chunk.content,
            score=item.score,
        )
        for rank, item in enumerate(scored, start=1)
    ]

    precision: float | None = None
    ndcg: float | None = None
    if body.expected_sources:
        paths = [chunk.source_path for chunk in chunks]
        precision = retrieval_precision(paths, body.expected_sources)
        ndcg = retrieval_ndcg(paths, body.expected_sources)

    return DebugQueryResponse(
        question=body.question,
        strategy=settings.retrieval_strategy,
        top_k=settings.retrieval_top_k,
        reranker=settings.reranker_provider,
        rrf_k=settings.rrf_k,
        chunks=chunks,
        stages=[stage.model_dump(mode="json") for stage in result.trace.stages],
        precision=precision,
        ndcg=ndcg,
    )


# --- Pipeline Trace: history ---

_PREVIEW_CHARS = 200


class InteractionListItem(BaseModel):
    """One past interaction in the trace history list."""

    id: str
    conversation_id: str
    question: str
    response_preview: str
    created_at: str
    has_enriched_trace: bool


class InteractionListResponse(BaseModel):
    """One page of past interactions, newest first, and how many match in all."""

    interactions: list[InteractionListItem]
    total: int


@router.get("/interactions", response_model=InteractionListResponse)
async def list_interactions(
    session: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    q: Annotated[str | None, Query(max_length=200)] = None,
) -> InteractionListResponse:
    """List past interactions, newest first, optionally only those whose question contains q.

    The match ignores case and treats % and _ in q as plain characters.
    response_preview is the first 200 characters of the response.
    has_enriched_trace is true when the interaction has a context trace row, which only
    interactions recorded with the enriched trace have.
    """
    term = (q or "").strip()
    conditions = [Interaction.question.icontains(term, autoescape=True)] if term else []

    # EXISTS rather than a join, so each interaction stays one row and pages stay exact.
    has_context_trace = (
        select(InteractionTrace.id)
        .where(
            InteractionTrace.interaction_id == Interaction.id,
            InteractionTrace.stage == _ENRICHED_STAGE,
        )
        .correlate(Interaction)
        .exists()
    )
    page_stmt = (
        select(
            Interaction.id,
            Interaction.conversation_id,
            Interaction.question,
            func.left(Interaction.response_text, _PREVIEW_CHARS).label("response_preview"),
            Interaction.created_at,
            has_context_trace.label("has_enriched_trace"),
        )
        .where(*conditions)
        .order_by(Interaction.created_at.desc(), Interaction.id.desc())
        .limit(limit)
        .offset(offset)
    )
    count_stmt = select(func.count()).select_from(Interaction).where(*conditions)

    total = (await session.execute(count_stmt)).scalar_one()
    rows = (await session.execute(page_stmt)).all()
    return InteractionListResponse(
        interactions=[
            InteractionListItem(
                id=str(row.id),
                conversation_id=str(row.conversation_id),
                question=row.question,
                response_preview=row.response_preview,
                created_at=row.created_at.isoformat(),
                has_enriched_trace=row.has_enriched_trace,
            )
            for row in rows
        ],
        total=total,
    )


class TraceDetailResponse(BaseModel):
    """Everything recorded for one interaction: the exchange, its stage traces, and its citations.

    interaction holds id, question, response_text, created_at, latency_ms, and token_count.
    traces always has retrieval, context, and generation keys, each None when that stage was not
    recorded, and holds the stored data of any other recorded stage.
    Each citation holds citation_index, chunk_id, source_path, content, score, and ranking_method.
    chunk_id matches the chunk_id in the retrieval stages' chunk_scores, and content is what the
    groundedness check needs.
    """

    interaction: dict
    traces: dict
    citations: list[dict]
    has_enriched_trace: bool


@router.get(
    "/traces/{interaction_id}",
    response_model=TraceDetailResponse,
    responses={404: {"description": "No such interaction"}},
)
async def get_trace(interaction_id: uuid.UUID, session: DbSession) -> TraceDetailResponse:
    """Return one past interaction with its recorded stage traces and citations."""
    interaction_stmt = select(
        Interaction.id,
        Interaction.question,
        Interaction.response_text,
        Interaction.created_at,
        Interaction.latency_ms,
        Interaction.token_count,
    ).where(Interaction.id == interaction_id)
    interaction = (await session.execute(interaction_stmt)).one_or_none()
    if interaction is None:
        raise HTTPException(status_code=404, detail="Interaction not found")

    trace_stmt = select(InteractionTrace.stage, InteractionTrace.data).where(
        InteractionTrace.interaction_id == interaction_id
    )
    citation_stmt = (
        select(
            InteractionCitation.citation_index,
            InteractionCitation.chunk_id,
            ChunkRecord.source_path,
            ChunkRecord.content,
            InteractionCitation.score,
            InteractionCitation.ranking_method,
        )
        .select_from(InteractionCitation)
        .join(ChunkRecord, ChunkRecord.chunk_id == InteractionCitation.chunk_id)
        .where(InteractionCitation.interaction_id == interaction_id)
        .order_by(InteractionCitation.citation_index)
    )
    trace_rows = (await session.execute(trace_stmt)).all()
    citation_rows = (await session.execute(citation_stmt)).all()

    recorded = {row.stage: row.data for row in trace_rows}
    return TraceDetailResponse(
        interaction={
            "id": str(interaction.id),
            "question": interaction.question,
            "response_text": interaction.response_text,
            "created_at": interaction.created_at.isoformat(),
            "latency_ms": interaction.latency_ms,
            "token_count": interaction.token_count,
        },
        traces={stage: recorded.get(stage) for stage in _TRACE_STAGES} | recorded,
        citations=[
            {
                "citation_index": row.citation_index,
                "chunk_id": row.chunk_id,
                "source_path": row.source_path,
                "content": row.content,
                "score": row.score,
                "ranking_method": row.ranking_method,
            }
            for row in citation_rows
        ],
        has_enriched_trace=_ENRICHED_STAGE in recorded,
    )


# --- Pipeline Trace: live ---


class LiveTraceRequest(BaseModel):
    """A question to run through the full pipeline, with optional earlier turns.

    Each history turn is a dict with a role ("user" or "assistant") and a content, oldest first,
    as in POST /chat.
    The pipeline copes with missing or unknown keys, so turns are not checked further.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)


class LiveTraceResponse(BaseModel):
    """The answer a question gets, as POST /ask returns it, plus the trace of how it was made.

    trace has retrieval, context, and generation keys, in the shape the history endpoint returns
    for an enriched interaction.
    """

    answer: str
    sources: list[dict]
    citations: list[dict]
    trace: dict


def _run_live_trace(request: Request, body: LiveTraceRequest) -> PipelineResult:
    """Run the full pipeline for the question while holding an LLM slot.

    Runs in a worker thread because retrieval and the LLM call block.
    """
    query = request.app.state.pipelines.query
    semaphore = acquire_llm_slot(request)
    try:
        return query.ask(body.question, history=body.history)
    finally:
        if semaphore is not None:
            semaphore.release()


async def _active_prompt_id(request: Request) -> int | None:
    """Return the id of the active system prompt version, or None when it cannot be read.

    The pipeline reads its prompt from PROMPT.md and does not say which saved version that is.
    PROMPT.md is written from the active version, so the active id normally names the prompt used.
    The lookup is best effort: a database that is off or failing must not stop a trace, so every
    failure is logged and gives None.
    """
    factory = getattr(request.app.state, "db_session_factory", None)
    if factory is None:
        return None
    try:
        async with factory() as session:
            result = await session.execute(
                select(PromptVersion.id).where(PromptVersion.is_active.is_(True)).limit(1)
            )
            return result.scalars().first()
    except Exception:
        logger.warning("Could not read the active system prompt id for a trace", exc_info=True)
        return None


@router.post(
    "/trace",
    response_model=LiveTraceResponse,
    dependencies=[Depends(require_retrieval_ready), Depends(require_generation_ready)],
    responses={503: {"description": "Retrieval or generation is not ready, or generation failed"}},
)
async def live_trace(request: Request, body: LiveTraceRequest) -> LiveTraceResponse:
    """Run the full pipeline for a question and return the answer with its trace.

    Nothing is saved, so the run leaves no interaction behind.
    The only database access is a read of the active prompt id, which is None when unavailable.
    """
    try:
        result = await asyncio.to_thread(_run_live_trace, request, body)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Live trace failed")
        raise HTTPException(status_code=503, detail="generation failed") from exc

    prompt_id = result.system_prompt_id
    if prompt_id is None:
        prompt_id = await _active_prompt_id(request)

    answer = result.answer
    return LiveTraceResponse(
        answer=answer.answer,
        sources=[source.model_dump(mode="json") for source in answer.sources],
        citations=[citation.model_dump(mode="json") for citation in answer.citations],
        trace={
            "retrieval": result.retrieval_trace.model_dump(mode="json"),
            "context": {
                "system_prompt_id": prompt_id,
                "history_turns": sum(1 for turn in body.history if turn.get("role") == "user"),
                "chunk_count": len(answer.sources),
                "messages": result.context_messages,
            },
            "generation": {
                **result.generation.model_dump(mode="json"),
                # The recorder stores this key too, so a live trace and a recorded one read alike.
                "generation_ms": answer.generation_ms,
            },
        },
    )


# --- Groundedness ---

# The most any one text field of a groundedness request may hold.
# It is above the longest answer the server can generate (LLM_MAX_TOKENS tops out at 16384
# tokens) and above any indexed chunk, so it only rejects payloads that cannot be real.
_MAX_FIELD_CHARS = 100_000


class GroundednessChunk(BaseModel):
    """One retrieved chunk to check the response against."""

    source_path: str = Field(max_length=_MAX_FIELD_CHARS)
    content: str = Field(max_length=_MAX_FIELD_CHARS)


class GroundednessRequest(BaseModel):
    """A response and the chunks it was generated from.

    The most chunks retrieval can return is TOP_K_MAX, so more than that cannot be real.
    An empty chunk list is allowed: every claim is then unsupported, which is the right verdict
    for an answer given with no context.
    """

    response_text: str = Field(min_length=1, max_length=_MAX_FIELD_CHARS)
    chunks: list[GroundednessChunk] = Field(max_length=TOP_K_MAX)


class GroundednessClaim(BaseModel):
    """One claim from the response and whether the chunks support it.

    A supported claim carries evidence and, when the judge named one, the chunk it came from:
    chunk_index is 1-based, in the order the chunks were sent, and source_path is that chunk's.
    An unsupported claim carries an explanation of what is missing.
    """

    claim: str
    verdict: Literal["supported", "unsupported"]
    evidence: str | None = None
    explanation: str | None = None
    chunk_index: int | None = None
    source_path: str | None = None


class GroundednessSummary(BaseModel):
    """Claim counts and the share of claims that are supported.

    score is supported divided by total_claims, from 0.0 to 1.0.
    It is 0.0 when there are no claims, so read total_claims before treating 0.0 as a failure.
    """

    total_claims: int
    supported: int
    unsupported: int
    score: float


class GroundednessResponse(BaseModel):
    """The claims found in a response, each with its verdict, and a summary of the verdicts."""

    claims: list[GroundednessClaim]
    summary: GroundednessSummary


def _ask_judge(request: Request, messages: list[dict[str, str]]) -> str:
    """Send messages to the judge LLM while holding an LLM slot, and return its reply text.

    The judge falls back to the server's own LLM, which may be one local model, so the call takes
    a slot, as the pipeline's own LLM calls do, instead of running beside a chat on that model.

    Raises:
        HTTPException: 503 when no judge model is configured or no LLM slot frees up in time.
    """
    try:
        judge = create_judge_client(request.app.state.settings)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    semaphore = acquire_llm_slot(request)
    try:
        return judge_chat(judge, messages)
    finally:
        if semaphore is not None:
            semaphore.release()


@router.post(
    "/groundedness",
    response_model=GroundednessResponse,
    responses={
        502: {"description": "The judge LLM replied, but not with a usable list of claims"},
        503: {"description": "The judge LLM is not configured, is busy, or failed"},
    },
)
def check_groundedness(request: Request, body: GroundednessRequest) -> GroundednessResponse:
    """Split a response into claims and judge each one against the chunks, using the judge LLM.

    Nothing is saved.
    Failures are not hidden as an empty result: a call that fails gives 503, and a reply that is
    not a usable claim list gives 502, so the caller can tell either from "no claims found".
    """
    chunks = [chunk.model_dump() for chunk in body.chunks]
    messages = build_groundedness_prompt(body.response_text, chunks)

    started = time.monotonic()
    try:
        reply = _ask_judge(request, messages)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Groundedness judge call failed")
        # The type name tells a timeout from a refused connection or a rejected request. The
        # message is left out because it can carry the judge's URL.
        raise HTTPException(
            status_code=503, detail=f"judge LLM request failed ({type(exc).__name__})"
        ) from exc
    elapsed_ms = int((time.monotonic() - started) * 1000)

    parsed = parse_groundedness_response(reply, chunks)
    if parsed.parse_error is not None:
        # The parser logs why it failed but not the reply, which only the caller holds.
        logger.warning("Unparseable groundedness reply (first 500 characters): %.500s", reply)
        raise HTTPException(
            status_code=502, detail=f"Judge LLM reply could not be parsed: {parsed.parse_error}"
        )

    try:
        response = GroundednessResponse(
            claims=[GroundednessClaim(**asdict(claim)) for claim in parsed.claims],
            summary=GroundednessSummary(
                total_claims=parsed.total_claims,
                supported=parsed.supported,
                unsupported=parsed.unsupported,
                score=parsed.score,
            ),
        )
    except ValidationError as exc:
        # The parser keeps evidence and explanation as the judge wrote them, which can be a list.
        logger.warning("Groundedness reply had claims in an unexpected format: %s", exc)
        raise HTTPException(
            status_code=502, detail="Judge LLM reply had claims in an unexpected format"
        ) from exc

    logger.info(
        "Groundedness check: %d of %d claims supported in %d ms",
        parsed.supported,
        parsed.total_claims,
        elapsed_ms,
    )
    return response
