"""add last_heartbeat_at, cancelled status, and update v_eval_run_summary

Revision ID: 004
Revises: 003
Create Date: 2026-10-21 18:26:03.825360
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "004"
down_revision: str = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "eval_runs",
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        schema="geoai",
    )
    op.add_column(
        "eval_runs",
        sa.Column("planned_question_ids", postgresql.JSONB, nullable=True),
        schema="geoai",
    )
    op.execute("ALTER TABLE geoai.eval_runs DROP CONSTRAINT IF EXISTS ck_eval_runs_status_valid")
    op.execute(
        "ALTER TABLE geoai.eval_runs ADD CONSTRAINT ck_eval_runs_status_valid "
        "CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'))"
    )
    op.execute("SET search_path TO geoai, public")
    op.execute("DROP VIEW IF EXISTS v_eval_run_summary")
    op.execute("""
        CREATE VIEW v_eval_run_summary AS
        SELECT er.id AS eval_run_id, er.eval_set_id, er.deployment_id,
               er.status, er.source,
               er.created_at,
               er.planned_count,
               er.last_heartbeat_at,
               d.label AS deployment_label,
               COUNT(res.id) AS questions_evaluated,
               AVG((res.scores->>'retrieval_precision')::float) AS avg_retrieval_precision,
               AVG((res.scores->>'retrieval_recall')::float) AS avg_retrieval_recall,
               AVG((res.scores->>'answer_similarity')::float) AS avg_answer_similarity,
               AVG((res.scores->>'faithfulness')::float) AS avg_faithfulness
        FROM eval_runs er
        JOIN deployments d ON d.id = er.deployment_id
        LEFT JOIN eval_results res ON res.eval_run_id = er.id
        GROUP BY er.id, d.id
    """)


def downgrade() -> None:
    op.execute("SET search_path TO geoai, public")
    op.execute("DROP VIEW IF EXISTS v_eval_run_summary")
    op.execute("""
        CREATE VIEW v_eval_run_summary AS
        SELECT er.id AS eval_run_id, er.eval_set_id, er.deployment_id,
               er.status, er.source,
               er.created_at,
               d.label AS deployment_label,
               COUNT(res.id) AS questions_evaluated,
               AVG((res.scores->>'retrieval_precision')::float) AS avg_retrieval_precision,
               AVG((res.scores->>'retrieval_recall')::float) AS avg_retrieval_recall,
               AVG((res.scores->>'answer_similarity')::float) AS avg_answer_similarity,
               AVG((res.scores->>'faithfulness')::float) AS avg_faithfulness
        FROM eval_runs er
        JOIN deployments d ON d.id = er.deployment_id
        LEFT JOIN eval_results res ON res.eval_run_id = er.id
        GROUP BY er.id, d.id
    """)
    op.execute("UPDATE geoai.eval_runs SET status = 'failed' WHERE status = 'cancelled'")
    op.execute("ALTER TABLE geoai.eval_runs DROP CONSTRAINT IF EXISTS ck_eval_runs_status_valid")
    op.execute(
        "ALTER TABLE geoai.eval_runs ADD CONSTRAINT ck_eval_runs_status_valid "
        "CHECK (status IN ('pending', 'running', 'completed', 'failed'))"
    )
    op.execute("ALTER TABLE geoai.eval_runs DROP COLUMN IF EXISTS planned_question_ids")
    op.drop_column("eval_runs", "last_heartbeat_at", schema="geoai")
