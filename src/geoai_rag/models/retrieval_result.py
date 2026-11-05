from __future__ import annotations

from pydantic import BaseModel

from geoai_rag.models.scored_chunk import ScoredChunk


class ChunkStageScore(BaseModel):
    """Score for one chunk at one retrieval stage."""

    chunk_id: str
    score: float


class StageTrace(BaseModel):
    """Trace data for one retrieval stage."""

    stage: str
    config: dict
    latency_ms: int
    input_count: int
    output_count: int
    chunk_scores: list[ChunkStageScore]


class RetrievalTrace(BaseModel):
    """Full trace across all retrieval stages."""

    stages: list[StageTrace]
    retrieval_query: str
    total_latency_ms: int


class RetrievalResult(BaseModel):
    """Retrieval output with both the ranked chunks and the pipeline trace."""

    chunks: list[ScoredChunk]
    trace: RetrievalTrace
