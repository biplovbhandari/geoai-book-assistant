from __future__ import annotations

from dataclasses import dataclass, field

from geoai_rag.models.answer import Answer
from geoai_rag.models.generation_result import GenerationResult
from geoai_rag.models.retrieval_result import RetrievalTrace


@dataclass
class PipelineResult:
    """Full pipeline output including the user-facing answer and trace data."""

    answer: Answer
    retrieval_trace: RetrievalTrace
    generation: GenerationResult
    context_messages: list[dict] = field(default_factory=list)
    system_prompt_id: int | None = None
