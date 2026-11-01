import hashlib
from datetime import datetime
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from geoai_rag.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from geoai_rag.db.models.evaluation import EvalRun
    from geoai_rag.db.models.interaction import Interaction, InteractionCitation


class PromptVersion(TimestampMixin, Base):
    """Content-addressed store of system prompt text."""

    __tablename__ = "prompt_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    hash: Mapped[str] = mapped_column(Text, unique=True)
    content: Mapped[str] = mapped_column(Text)
    label: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(sa.Boolean, default=False)

    deployments: Mapped[list["Deployment"]] = relationship(
        back_populates="prompt_version", passive_deletes="all"
    )

    @staticmethod
    def compute_hash(content: str) -> str:
        """SHA-256 hex digest of the prompt text, stripped of surrounding whitespace.

        Stripping ensures the hash is stable regardless of trailing newlines in PROMPT.md.
        All callers must use this method to avoid hash mismatches.
        """
        return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


class IndexRun(TimestampMixin, Base):
    """Immutable record of each book indexing event."""

    __tablename__ = "index_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    commit_sha: Mapped[str] = mapped_column(Text)
    config: Mapped[Any] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    chunks: Mapped[list["ChunkRecord"]] = relationship(
        back_populates="index_run", passive_deletes="all"
    )
    deployments: Mapped[list["Deployment"]] = relationship(
        back_populates="index_run", passive_deletes="all"
    )


class Deployment(TimestampMixin, Base):
    """Immutable snapshot of a complete pipeline configuration."""

    __tablename__ = "deployments"
    __table_args__ = (
        CheckConstraint("retrieval_strategy IN ('dense', 'hybrid')", name="strategy_valid"),
        Index(
            "idx_deployments_is_active",
            "is_active",
            unique=True,
            postgresql_where=text("is_active IS TRUE"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    prompt_version_id: Mapped[int] = mapped_column(
        ForeignKey("prompt_versions.id", ondelete="RESTRICT")
    )
    index_run_id: Mapped[int] = mapped_column(ForeignKey("index_runs.id", ondelete="RESTRICT"))
    model_name: Mapped[str] = mapped_column(Text)
    temperature: Mapped[float]
    retrieval_strategy: Mapped[str] = mapped_column(Text)
    label: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool]

    prompt_version: Mapped["PromptVersion"] = relationship(back_populates="deployments")
    index_run: Mapped["IndexRun"] = relationship(back_populates="deployments")
    interactions: Mapped[list["Interaction"]] = relationship(
        back_populates="deployment", passive_deletes="all"
    )
    eval_runs: Mapped[list["EvalRun"]] = relationship(
        back_populates="deployment", passive_deletes="all"
    )


class ChunkRecord(Base):
    """A single indexed passage from the book."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    index_run_id: Mapped[int] = mapped_column(
        ForeignKey("index_runs.id", ondelete="RESTRICT"), index=True
    )
    content: Mapped[str] = mapped_column(Text)
    source_path: Mapped[str] = mapped_column(Text, index=True)
    chapter: Mapped[str | None] = mapped_column(Text)
    section: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    display_label: Mapped[str | None] = mapped_column(Text)

    index_run: Mapped["IndexRun"] = relationship(back_populates="chunks")
    citations: Mapped[list["InteractionCitation"]] = relationship(
        back_populates="chunk", passive_deletes="all"
    )
