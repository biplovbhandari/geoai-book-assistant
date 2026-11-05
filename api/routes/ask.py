from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from api.dependencies import acquire_llm_slot, require_generation_ready, require_retrieval_ready
from api.metrics import llm_generation_seconds, retrieval_seconds
from api.routes.validators import QuestionFiltersMixin
from geoai_rag.models.answer import Answer

logger = logging.getLogger(__name__)

router = APIRouter()


class AskRequest(QuestionFiltersMixin):
    """Incoming question with optional metadata filters."""


@router.post(
    "/ask",
    response_model=Answer,
    dependencies=[Depends(require_retrieval_ready), Depends(require_generation_ready)],
)
def ask(request: Request, body: AskRequest):
    """Generate an answer from the book using retrieval-augmented generation.

    Returns 503 if pipelines are not ready or if generation fails.
    """
    pipelines = request.app.state.pipelines

    llm_semaphore = acquire_llm_slot(request)
    try:
        pipeline_result = pipelines.query.ask(body.question, body.filters)
        retrieval_seconds.observe(pipeline_result.answer.retrieval_ms / 1000)
        llm_generation_seconds.observe(pipeline_result.answer.generation_ms / 1000)
    except Exception:
        logger.exception("Generation failed")
        raise HTTPException(status_code=503, detail="generation failed")
    finally:
        if llm_semaphore is not None:
            llm_semaphore.release()

    # Return only the Answer: the trace data on PipelineResult is not part of the widget contract.
    return pipeline_result.answer
