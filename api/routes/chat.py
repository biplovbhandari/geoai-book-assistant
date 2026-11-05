from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from starlette.background import BackgroundTask

from api.dependencies import acquire_llm_slot, require_generation_ready, require_retrieval_ready
from api.metrics import llm_generation_seconds, retrieval_seconds
from api.recording import RecordingContext, record_interaction
from api.routes.validators import QuestionFiltersMixin
from geoai_rag.db.models.infrastructure import PromptVersion

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_CONTENT = 8000


class ChatMessage(BaseModel):
    """A single conversation turn in the chat history."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(QuestionFiltersMixin):
    """Incoming chat question with optional history and filters."""

    history: list[ChatMessage] = Field(default_factory=list)
    conversation_id: uuid.UUID | None = None
    visitor_id: uuid.UUID | None = None
    ga4_client_id: str | None = Field(default=None, max_length=64)

    @field_validator("history", mode="before")
    @classmethod
    def truncate_history(cls, v):
        """Truncate history: cap count, cap per-entry content, drop orphan leading assistant."""
        if not isinstance(v, list):
            return v
        if len(v) > MAX_HISTORY_MESSAGES:
            v = v[-MAX_HISTORY_MESSAGES:]
        for entry in v:
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("content"), str)
                and len(entry["content"]) > MAX_HISTORY_CONTENT
            ):
                entry["content"] = entry["content"][:MAX_HISTORY_CONTENT]
        while v and isinstance(v[0], dict) and v[0].get("role") == "assistant":
            v = v[1:]
        return v


@router.post(
    "/chat", dependencies=[Depends(require_retrieval_ready), Depends(require_generation_ready)]
)
def chat(request: Request, body: ChatRequest):
    """Stream a RAG answer as server-sent events, recording it in the background.

    Returns 503 if pipelines are not ready or streaming is unsupported. Each SSE
    event is a JSON-encoded dict emitted by the query pipeline's ask_stream method;
    interaction/conversation/visitor IDs are injected into the ``meta`` event per
    the API contract. After the stream completes, a BackgroundTask embeds the
    retrieval query and persists the interaction -- this runs after the response
    has been sent, so a recording failure never affects the delivered answer.
    """
    pipelines = request.app.state.pipelines
    if not callable(getattr(pipelines.query._llm_client, "chat_stream", None)):
        raise HTTPException(status_code=503, detail="streaming not supported")

    interaction_id = uuid.uuid4()
    conversation_id = body.conversation_id or uuid.uuid4()
    visitor_id = body.visitor_id or uuid.uuid4()
    is_new_conversation = body.conversation_id is None

    # Recording prerequisites -- all four must be present for the background
    # task to persist anything. Missing any one (e.g. DB down, no active
    # deployment) degrades to "don't record" rather than failing the request.
    deployment_id = getattr(request.app.state, "active_deployment_id", None)
    index_run_id = getattr(request.app.state, "active_index_run_id", None)
    session_factory = getattr(request.app.state, "db_session_factory", None)
    embedder = getattr(pipelines, "embedder", None)
    can_record = all([session_factory, deployment_id, index_run_id, embedder])

    settings = getattr(request.app.state, "settings", None)
    ctx = RecordingContext(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        visitor_id=visitor_id,
        is_new_conversation=is_new_conversation,
        ga4_client_id=body.ga4_client_id,
        question=body.question,
        model_name=settings.llm_model if settings else "",
    )
    pipeline_ctx: dict = {}  # populated by ask_stream via _recording_ctx

    history_dicts = [{"role": m.role, "content": m.content} for m in body.history]
    ctx.history_turn_count = len(history_dicts)

    llm_semaphore = acquire_llm_slot(request)

    def event_stream():
        """Yield SSE-formatted events from the streaming pipeline.

        Accumulates response text, token count, and completion state onto
        ``ctx`` as events pass through, so the background recording task has
        everything it needs once the stream is exhausted.
        """
        start_time = time.monotonic()
        try:
            for event in pipelines.query.ask_stream(
                body.question,
                history=history_dicts,
                filters=body.filters,
                _recording_ctx=pipeline_ctx,
            ):
                if event["type"] == "meta":
                    # Inject IDs per API contract
                    event["interaction_id"] = str(interaction_id)
                    if is_new_conversation:
                        event["conversation_id"] = str(conversation_id)
                    if body.visitor_id is None:
                        event["visitor_id"] = str(visitor_id)
                elif event["type"] == "token":
                    ctx.response_text += event.get("content", "")
                    ctx.token_event_count += 1
                elif event["type"] == "done":
                    ctx.saw_done = True
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            logger.exception("Streaming generation failed")
            error = {"type": "error", "message": "Generation failed. Please try again."}
            yield f"data: {json.dumps(error)}\n\n"

        ctx.latency_ms = int((time.monotonic() - start_time) * 1000)
        ctx.scored_chunks = pipeline_ctx.get("scored_chunks", [])
        ctx.retrieval_query = pipeline_ctx.get("retrieval_query", "")
        ctx.retrieval_ms = pipeline_ctx.get("retrieval_ms", 0)
        ctx.generation_wall_ms = pipeline_ctx.get("generation_wall_ms", 0)
        # These keys are absent when no chunks were found (generation skipped) or when the stream
        # reported no metadata. None makes the recorder fall back to its minimal row data.
        ctx.retrieval_trace = pipeline_ctx.get("retrieval_trace")
        ctx.generation_metadata = pipeline_ctx.get("generation_metadata")
        ctx.context_messages = pipeline_ctx.get("context_messages")

    async def _active_prompt_id() -> int | None:
        """Return the active system prompt's id, or None when it cannot be read.

        Recording must never fail because of this lookup, so any error degrades to None.
        """
        if session_factory is None:
            return None
        try:
            async with session_factory() as session:
                result = await session.execute(
                    select(PromptVersion.id).where(PromptVersion.is_active.is_(True))
                )
                return result.scalars().first()
        except Exception:
            logger.warning("Could not read the active system prompt id", exc_info=True)
            return None

    async def _cleanup():
        """Release the LLM semaphore and persist the interaction. Never raises.

        Runs as a StreamingResponse BackgroundTask after the SSE response
        has been fully sent (or the client has disconnected). The semaphore
        is released here instead of inside the generator so that client
        disconnects cannot leak the slot.
        """
        try:
            if llm_semaphore is not None:
                llm_semaphore.release()
        except Exception:
            logger.exception("Failed to release LLM semaphore for %s", ctx.interaction_id)
        if ctx.retrieval_ms:
            retrieval_seconds.observe(ctx.retrieval_ms / 1000)
        if ctx.generation_wall_ms:
            llm_generation_seconds.observe(ctx.generation_wall_ms / 1000)
        try:
            if not (can_record and ctx.saw_done):
                return
            assert embedder is not None
            assert session_factory is not None
            assert deployment_id is not None
            assert index_run_id is not None
            ctx.system_prompt_id = await _active_prompt_id()
            embedding = await asyncio.to_thread(embedder.embed_query, ctx.retrieval_query)
            await record_interaction(
                session_factory,
                ctx,
                embedding=embedding,
                deployment_id=deployment_id,
                index_run_id=index_run_id,
            )
        except Exception:
            logger.exception("Background recording failed for %s", ctx.interaction_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        background=BackgroundTask(_cleanup),
    )
