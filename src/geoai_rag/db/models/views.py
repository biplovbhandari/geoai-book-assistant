"""Read-only ORM models for database views.

These map to views created by Alembic migrations (raw SQL).
They enable typed queries and auto-generated schema diagrams.
They do NOT create the views - migrations do that.
"""

import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import DateTime, Float, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from geoai_rag.db.base import Base

_VIEW = {"info": {"is_view": True}}


class VInteractionSummary(Base):
    """Per-interaction detail with feedback and deployment context."""

    __tablename__ = "v_interaction_summary"
    __table_args__ = _VIEW

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    response_text: Mapped[str] = mapped_column(Text)
    latency_ms: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    rating: Mapped[str | None] = mapped_column(Text)
    comment: Mapped[str | None] = mapped_column(Text)
    admin_tags: Mapped[Any | None] = mapped_column(Text)
    citation_count: Mapped[int] = mapped_column(Integer)
    deployment_label: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(Text)
    temperature: Mapped[float] = mapped_column(Float)


class VDailyStats(Base):
    """Aggregated daily interaction statistics."""

    __tablename__ = "v_daily_stats"
    __table_args__ = _VIEW

    day: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    total_queries: Mapped[int] = mapped_column(Integer)
    thumbs_up: Mapped[int] = mapped_column(Integer)
    thumbs_down: Mapped[int] = mapped_column(Integer)
    avg_tokens: Mapped[float | None] = mapped_column(Float)
    avg_latency_ms: Mapped[float | None] = mapped_column(Float)


class VCitationHeatmap(Base):
    """Citation frequency and satisfaction by source path."""

    __tablename__ = "v_citation_heatmap"
    __table_args__ = _VIEW

    source_path: Mapped[str] = mapped_column(Text, primary_key=True)
    display_label: Mapped[str | None] = mapped_column(Text)
    cite_count: Mapped[int] = mapped_column(Integer)
    thumbs_up_pct: Mapped[float | None] = mapped_column(Float)
    last_cited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ranking_methods: Mapped[str | None] = mapped_column(Text)


class VRetrievalGaps(Base):
    """Questions with low retrieval scores and negative feedback."""

    __tablename__ = "v_retrieval_gaps"
    __table_args__ = _VIEW

    question: Mapped[str] = mapped_column(Text, primary_key=True)
    avg_top_score: Mapped[float | None] = mapped_column(Float)
    thumbs_down_count: Mapped[int] = mapped_column(Integer)
    ranking_methods: Mapped[str | None] = mapped_column(Text)


class VDeploymentMetrics(Base):
    """Per-deployment interaction counts and satisfaction."""

    __tablename__ = "v_deployment_metrics"
    __table_args__ = _VIEW

    deployment_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(Text)
    retrieval_strategy: Mapped[str] = mapped_column(Text)
    interaction_count: Mapped[int] = mapped_column(Integer)
    satisfaction_rate: Mapped[float | None] = mapped_column(Float)
    avg_latency_ms: Mapped[float | None] = mapped_column(Float)


class VConversationSummary(Base):
    """Per-conversation aggregates: turn count, last activity, duration."""

    __tablename__ = "v_conversation_summary"
    __table_args__ = _VIEW

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    title: Mapped[str | None] = mapped_column(Text)
    topic: Mapped[str | None] = mapped_column(Text)
    visitor_id: Mapped[uuid.UUID | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    interaction_count: Mapped[int] = mapped_column(Integer)
    last_activity: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration: Mapped[timedelta | None]


class VEvalRunSummary(Base):
    """Per-eval-run aggregated scores from eval_results."""

    __tablename__ = "v_eval_run_summary"
    __table_args__ = _VIEW

    eval_run_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    eval_set_id: Mapped[int] = mapped_column(Integer)
    deployment_id: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    planned_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    deployment_label: Mapped[str | None] = mapped_column(Text)
    questions_evaluated: Mapped[int] = mapped_column(Integer)
    avg_retrieval_precision: Mapped[float | None] = mapped_column(Float)
    avg_retrieval_recall: Mapped[float | None] = mapped_column(Float)
    avg_answer_similarity: Mapped[float | None] = mapped_column(Float)
    avg_faithfulness: Mapped[float | None] = mapped_column(Float)
