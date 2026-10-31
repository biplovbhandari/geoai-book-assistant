from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from geoai_rag.config import Settings
from geoai_rag.db.models.infrastructure import Deployment, IndexRun, PromptVersion
from geoai_rag.generation.context_builder import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Matches OpenAICompatibleClient.chat()/chat_stream() default temperature.
# Settings has no temperature field, so the deployment snapshot hardcodes it.
_DEFAULT_TEMPERATURE = 0.3


async def ensure_active_deployment(
    session: AsyncSession,
    settings: Settings,
) -> tuple[int, int] | None:
    """Ensure an active Deployment row exists, creating the full chain if needed.

    Looks up the deployment with ``is_active IS TRUE``. If none exists yet,
    creates a PromptVersion (content-addressed by SHA-256 of PROMPT.md), an
    IndexRun (config snapshot from settings), and a Deployment that claims
    the single active-deployment slot -- all inside a savepoint, so that a
    concurrent caller winning the race leaves no orphaned rows behind.

    Intended to run once at app startup so chat recording has a
    deployment_id/index_run_id to attach to every interaction. Never raises:
    any failure is logged and reported as None so the app can still start
    with recording disabled.

    Args:
        session: An open AsyncSession. On the lost-race path this function
            does not call session.commit() (nothing of ours was persisted);
            the caller owns closing/rolling back the session.
        settings: Application settings supplying the model/prompt/config.

    Returns:
        (deployment_id, index_run_id) on success, None on failure.
    """
    try:
        existing = await _get_active_deployment(session)
        if existing is not None:
            return existing

        savepoint = await session.begin_nested()

        prompt_hash, prompt_content = _read_prompt()
        prompt_version_id = await _upsert_prompt_version(session, prompt_hash, prompt_content)
        index_run_id = await _insert_index_run(session, settings)
        claimed = await _insert_deployment(session, settings, prompt_version_id, index_run_id)

        if claimed is None:
            # Lost the race for the single active-deployment slot: discard our
            # orphaned IndexRun/PromptVersion writes and defer to the winner.
            await savepoint.rollback()
            return await _get_active_deployment(session)

        # Fold the savepoint's writes into the outer transaction, then commit
        # the outer transaction so the caller gets ids that are actually durable.
        await savepoint.commit()
        await session.commit()
        return claimed
    except Exception:
        logger.warning(
            "Failed to ensure active deployment; chat recording disabled",
            exc_info=True,
        )
        return None


async def _get_active_deployment(session: AsyncSession) -> tuple[int, int] | None:
    """Return (deployment_id, index_run_id) for the active deployment, or None."""
    stmt = select(Deployment.id, Deployment.index_run_id).where(Deployment.is_active.is_(True))
    row = (await session.execute(stmt)).first()
    if row is None:
        return None
    return (row[0], row[1])


def _read_prompt() -> tuple[str, str]:
    """Return (sha256_hexdigest, content) of the system prompt."""
    prompt_hash = PromptVersion.compute_hash(SYSTEM_PROMPT)
    return prompt_hash, SYSTEM_PROMPT


async def _upsert_prompt_version(session: AsyncSession, prompt_hash: str, content: str) -> int:
    """Insert a PromptVersion if its hash is new, then return its id either way."""
    insert_stmt = (
        insert(PromptVersion)
        .values(hash=prompt_hash, content=content)
        .on_conflict_do_nothing(index_elements=["hash"])
    )
    await session.execute(insert_stmt)

    id_stmt = select(PromptVersion.id).where(PromptVersion.hash == prompt_hash)
    return (await session.execute(id_stmt)).scalar_one()


def _build_config_snapshot(settings: Settings) -> dict[str, Any]:
    """Build the IndexRun config JSONB payload.

    Explicit allowlist -- must never include secrets such as database_url
    or llm_api_key.
    """
    return {
        "embedding_model_name": settings.embedding_model_name,
        "retrieval_strategy": settings.retrieval_strategy,
        "llm_model": settings.llm_model,
        "reranker_provider": settings.reranker_provider,
        "sparse_model_name": settings.sparse_model_name,
    }


async def _insert_index_run(session: AsyncSession, settings: Settings) -> int:
    """Insert a new IndexRun row and return its id."""
    stmt = (
        insert(IndexRun)
        .values(commit_sha=settings.book_commit_sha, config=_build_config_snapshot(settings))
        .returning(IndexRun.id)
    )
    return (await session.execute(stmt)).scalar_one()


async def _insert_deployment(
    session: AsyncSession,
    settings: Settings,
    prompt_version_id: int,
    index_run_id: int,
) -> tuple[int, int] | None:
    """Try to insert a new active Deployment, claiming the single active slot.

    Uses the partial unique index on is_active to atomically claim the slot;
    a concurrent transaction that wins the race causes this insert to be a
    no-op (on_conflict_do_nothing) instead of raising a constraint violation.

    Returns:
        (deployment_id, index_run_id) if this insert claimed the slot,
        None if a concurrent transaction already holds it.
    """
    stmt = (
        insert(Deployment)
        .values(
            prompt_version_id=prompt_version_id,
            index_run_id=index_run_id,
            model_name=settings.llm_model,
            temperature=_DEFAULT_TEMPERATURE,
            retrieval_strategy=settings.retrieval_strategy,
            is_active=True,
        )
        .on_conflict_do_nothing(
            index_elements=["is_active"],
            index_where=text("is_active IS TRUE"),
        )
        .returning(Deployment.id, Deployment.index_run_id)
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        return None
    return (row[0], row[1])
