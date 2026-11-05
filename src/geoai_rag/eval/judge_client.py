from __future__ import annotations

from geoai_rag.config import Settings
from geoai_rag.eval.runner import JUDGE_TIMEOUT_SECONDS
from geoai_rag.generation.llm_client import OpenAICompatibleClient


def create_judge_client(
    settings: Settings, default_model: str | None = None
) -> OpenAICompatibleClient:
    """Build an LLM client for evaluation judging from the EVAL_JUDGE_* settings.

    Each judge setting falls back to the matching server LLM setting when empty. The model
    falls back to *default_model* first (for callers that learned the server's model from
    the running API), then to LLM_MODEL. The timeout is the eval runner's, because a
    reasoning model can think for minutes.

    Args:
        settings: Application settings.
        default_model: Model to use when EVAL_JUDGE_MODEL is empty, before LLM_MODEL.

    Returns:
        A client for the judge LLM.

    Raises:
        ValueError: When no model is named by EVAL_JUDGE_MODEL, *default_model*, or LLM_MODEL.
    """
    model = settings.eval_judge_model or default_model or settings.llm_model
    if not model:
        raise ValueError("No judge model configured; set EVAL_JUDGE_MODEL or LLM_MODEL")
    return OpenAICompatibleClient(
        base_url=settings.eval_judge_base_url or settings.llm_base_url,
        api_key=(
            settings.eval_judge_api_key.get_secret_value()
            or settings.llm_api_key.get_secret_value()
        ),
        model=model,
        timeout=JUDGE_TIMEOUT_SECONDS,
        reasoning_effort=settings.eval_judge_reasoning_effort,
    )
