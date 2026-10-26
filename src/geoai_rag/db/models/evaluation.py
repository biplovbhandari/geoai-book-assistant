from datetime import datetime
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import CheckConstraint, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from geoai_rag.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from geoai_rag.db.models.infrastructure import Deployment


class EvalSet(TimestampMixin, Base):
    """A named collection of test questions for evaluating pipeline quality."""

    __tablename__ = "eval_sets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)

    questions: Mapped[list["EvalQuestion"]] = relationship(
        back_populates="eval_set", cascade="all, delete", passive_deletes=True
    )
    runs: Mapped[list["EvalRun"]] = relationship(back_populates="eval_set", passive_deletes="all")


class EvalQuestion(Base):
    """One test question with ground truth for evaluation."""

    __tablename__ = "eval_questions"
    __table_args__ = (
        UniqueConstraint("eval_set_id", "question_key", name="uq_eval_questions_set_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    eval_set_id: Mapped[int] = mapped_column(
        ForeignKey("eval_sets.id", ondelete="CASCADE"), index=True
    )
    # The question's id in the eval set YAML (q001). Unlike the text, it survives rewording.
    # NULL for rows saved before this column existed; NULLs never clash in the unique constraint.
    question_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    question: Mapped[str] = mapped_column(Text)
    expected_answer: Mapped[str | None] = mapped_column(Text)
    expected_sources: Mapped[Any] = mapped_column(JSONB)
    tags: Mapped[Any] = mapped_column(JSONB, nullable=True)

    eval_set: Mapped["EvalSet"] = relationship(back_populates="questions")
    results: Mapped[list["EvalResult"]] = relationship(
        back_populates="question", passive_deletes=True
    )


class EvalRun(TimestampMixin, Base):
    """One execution of an eval set against a specific deployment."""

    __tablename__ = "eval_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="status_valid",
        ),
        CheckConstraint(
            "source IN ('cli', 'api')",
            name="source_valid",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    eval_set_id: Mapped[int] = mapped_column(
        ForeignKey("eval_sets.id", ondelete="RESTRICT"), index=True
    )
    deployment_id: Mapped[int] = mapped_column(
        ForeignKey("deployments.id", ondelete="RESTRICT"), index=True
    )
    status: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text, default="cli")
    config: Mapped[Any] = mapped_column(JSONB, nullable=True)
    planned_count: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    planned_question_ids: Mapped[list[int] | None] = mapped_column(JSONB, nullable=True)

    eval_set: Mapped["EvalSet"] = relationship(back_populates="runs")
    deployment: Mapped["Deployment"] = relationship(back_populates="eval_runs")
    results: Mapped[list["EvalResult"]] = relationship(
        back_populates="eval_run", cascade="all, delete", passive_deletes=True
    )


class EvalResult(Base):
    """The actual pipeline output and scores for one question in one run."""

    __tablename__ = "eval_results"
    __table_args__ = (UniqueConstraint("eval_run_id", "question_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    eval_run_id: Mapped[int] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True
    )
    question_id: Mapped[int | None] = mapped_column(
        ForeignKey("eval_questions.id", ondelete="SET NULL"), index=True
    )
    question_snapshot: Mapped[str] = mapped_column(Text)
    expected_answer_snapshot: Mapped[str | None] = mapped_column(Text)
    expected_sources_snapshot: Mapped[Any] = mapped_column(JSONB)
    response: Mapped[str] = mapped_column(Text)
    citations: Mapped[Any] = mapped_column(JSONB)
    trace: Mapped[Any] = mapped_column(JSONB)
    scores: Mapped[Any] = mapped_column(JSONB)

    eval_run: Mapped["EvalRun"] = relationship(back_populates="results")
    question: Mapped["EvalQuestion | None"] = relationship(back_populates="results")
