"""Admin debug endpoint: run retrieval only and show the ranked chunks.

Used by the dashboard's Pipeline Inspector to see what the retriever returns for a question,
without paying for a generation.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.dependencies import require_admin_token, require_retrieval_ready
from geoai_rag.eval.scorer import retrieval_ndcg, retrieval_precision

logger = logging.getLogger(__name__)

router = APIRouter(tags=["debug"], dependencies=[Depends(require_admin_token)])


class DebugQueryRequest(BaseModel):
    """A question to retrieve for, with optional expected source prefixes to score against."""

    question: str = Field(min_length=1, max_length=2000)
    expected_sources: list[str] | None = None


class ChunkResult(BaseModel):
    """One retrieved chunk."""

    rank: int
    source_path: str
    text: str
    # Only the final score is available. Per-stage scores (semantic, BM25, RRF, reranker) need
    # instrumentation in the retrieval strategies before they can be shown here.
    score: float


class DebugQueryResponse(BaseModel):
    """Retrieval settings in effect, the ranked chunks, and optional precision and NDCG."""

    question: str
    strategy: str
    top_k: int
    reranker: str
    rrf_k: int
    chunks: list[ChunkResult]
    precision: float | None
    ndcg: float | None


@router.post(
    "/query", response_model=DebugQueryResponse, dependencies=[Depends(require_retrieval_ready)]
)
def debug_query(request: Request, body: DebugQueryRequest) -> DebugQueryResponse:
    """Run retrieval (no generation) and return the ranked chunks."""
    settings = request.app.state.settings
    pipelines = request.app.state.pipelines

    try:
        scored = pipelines.query.search(body.question, settings.retrieval_top_k)
    except Exception:
        logger.exception("Debug retrieval failed")
        raise HTTPException(status_code=503, detail="retrieval not ready")

    chunks = [
        ChunkResult(
            rank=rank,
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
        precision=precision,
        ndcg=ndcg,
    )
