"""Admin endpoints for versioned system prompts.

Every version of the system prompt is a row in prompt_versions, and at most one row is active.
Rows are content-addressed: saving text that is already stored reuses its row.
Creating or activating a version also writes its text to PROMPT.md, the file the pipeline reads.
That write is best-effort: the database row is the record, and a read-only file system must not
fail the request.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi import Path as PathParam
from pydantic import BaseModel, Field
from sqlalchemy import select, update

from api.dependencies import DbSession, require_admin_token
from geoai_rag.db.models import PromptVersion

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parents[2] / "src" / "geoai_rag" / "generation" / "PROMPT.md"

# The largest value of a Postgres integer column; a bigger id could only fail inside the database.
_MAX_ID = 2**31 - 1
_Id = Annotated[int, PathParam(ge=1, le=_MAX_ID)]

router = APIRouter(tags=["prompts"], dependencies=[Depends(require_admin_token)])


def compute_prompt_hash(content: str) -> str:
    """Delegate to PromptVersion.compute_hash for backward compatibility."""
    return PromptVersion.compute_hash(content)


def write_prompt_file(content: str) -> None:
    """Write the prompt text to PROMPT.md, logging a warning instead of raising on failure."""
    try:
        PROMPT_PATH.write_text(content, encoding="utf-8")
    except OSError:
        logger.warning("Could not write system prompt to %s", PROMPT_PATH, exc_info=True)


class PromptSummary(BaseModel):
    """One prompt version without its text, which can be large."""

    id: int
    hash: str
    label: str | None
    is_active: bool
    created_at: datetime


class PromptDetail(PromptSummary):
    """One prompt version with its full text."""

    content: str


class PromptCreate(BaseModel):
    """Body of POST /prompts: the new prompt text and an optional label."""

    content: str = Field(min_length=1)
    label: str | None = None


class PromptCreated(BaseModel):
    """The id and hash of a prompt version that is now active."""

    id: int
    hash: str


class PromptActivated(BaseModel):
    """The id and hash of the prompt version that is now active."""

    id: int
    hash: str


async def _deactivate_all(session: DbSession) -> None:
    """Mark every prompt version inactive."""
    await session.execute(update(PromptVersion).values(is_active=False))


@router.get("", response_model=list[PromptSummary])
async def list_prompts(session: DbSession) -> list[PromptVersion]:
    """List every prompt version, newest first, without the prompt text."""
    result = await session.execute(select(PromptVersion).order_by(PromptVersion.created_at.desc()))
    return list(result.scalars().all())


@router.get("/active", response_model=PromptDetail)
async def get_active_prompt(session: DbSession) -> PromptVersion:
    """Return the active prompt version with its text, or 404 if none is active."""
    result = await session.execute(select(PromptVersion).where(PromptVersion.is_active.is_(True)))
    active = result.scalars().first()
    if active is None:
        raise HTTPException(status_code=404, detail="No active system prompt")
    return active


@router.post("", response_model=PromptCreated, status_code=201)
async def create_prompt(
    body: PromptCreate, session: DbSession, response: Response
) -> PromptCreated:
    """Make the given text the only active prompt version and write it to PROMPT.md.

    Returns 201 for a new row. If a row with the same hash exists it is reused (and relabelled when
    a label is given) and the response is 200, so identical text never creates a duplicate.
    """
    prompt_hash = compute_prompt_hash(body.content)
    found = await session.execute(select(PromptVersion).where(PromptVersion.hash == prompt_hash))
    row = found.scalars().first()
    await _deactivate_all(session)
    if row is None:
        row = PromptVersion(
            hash=prompt_hash, content=body.content, label=body.label, is_active=True
        )
        session.add(row)
        await session.flush()
    else:
        row.is_active = True
        if body.label is not None:
            row.label = body.label
        response.status_code = 200
    await session.commit()
    write_prompt_file(body.content)
    return PromptCreated(id=row.id, hash=row.hash)


@router.post("/{prompt_id}/activate", response_model=PromptActivated)
async def activate_prompt(prompt_id: _Id, session: DbSession) -> PromptActivated:
    """Make an older prompt version the only active one and write it to PROMPT.md."""
    row = await session.get(PromptVersion, prompt_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"System prompt {prompt_id} not found")
    await _deactivate_all(session)
    row.is_active = True
    await session.commit()
    write_prompt_file(row.content)
    return PromptActivated(id=row.id, hash=row.hash)
