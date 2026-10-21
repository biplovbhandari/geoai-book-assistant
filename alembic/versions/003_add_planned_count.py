"""add planned_count

Revision ID: 003
Revises: 002
Create Date: 2026-10-21 16:09:35.830872
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "003"
down_revision: str = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "eval_runs", sa.Column("planned_count", sa.Integer(), nullable=True), schema="geoai"
    )


def downgrade() -> None:
    op.drop_column("eval_runs", "planned_count", schema="geoai")
