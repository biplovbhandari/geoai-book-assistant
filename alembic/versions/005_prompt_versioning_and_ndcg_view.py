"""add prompt_versions.is_active and eval_runs.system_prompt_id, rename view column to avg_retrieval_ndcg

The eval runner now stores NDCG under scores.retrieval_ndcg. Older rows keep their
retrieval_recall key, so the view reads only the new key and old runs show no NDCG.

Revision ID: 005
Revises: 004
Create Date: 2026-10-31 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "005"
down_revision: str = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "prompt_versions",
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.false()),
        schema="geoai",
    )
    op.add_column(
        "eval_runs",
        sa.Column("system_prompt_id", sa.Integer, nullable=True),
        schema="geoai",
    )
    op.create_foreign_key(
        "fk_eval_runs_system_prompt_id",
        "eval_runs",
        "prompt_versions",
        ["system_prompt_id"],
        ["id"],
        source_schema="geoai",
        referent_schema="geoai",
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
               AVG((res.scores->>'retrieval_ndcg')::float) AS avg_retrieval_ndcg,
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
    op.drop_constraint(
        "fk_eval_runs_system_prompt_id", "eval_runs", schema="geoai", type_="foreignkey"
    )
    op.drop_column("eval_runs", "system_prompt_id", schema="geoai")
    op.drop_column("prompt_versions", "is_active", schema="geoai")
