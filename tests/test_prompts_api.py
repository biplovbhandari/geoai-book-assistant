"""Tests for the /prompts endpoints and the startup seeding of the first prompt version.

The database is a canned FakeSession, so no PostgreSQL is needed.
PROMPT.md is redirected to a temporary file so no test touches the real prompt.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import create_test_client

from api.dependencies import Pipelines, require_admin_token, require_db_session
from api.main import _seed_system_prompt
from api.routes import prompts as prompts_routes
from geoai_rag.db.models import PromptVersion

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)


class FakeResult:
    """The slice of a SQLAlchemy Result the prompt endpoints call: scalars().all() and first()."""

    def __init__(self, rows: list[Any]) -> None:
        """Hold the rows a statement found."""
        self._rows = rows

    def scalars(self) -> FakeResult:
        """Return self, because the endpoints only call all() and first() on the scalars."""
        return self

    def all(self) -> list[Any]:
        """Return every row."""
        return self._rows

    def first(self) -> Any:
        """Return the first row, or None when there are none."""
        return self._rows[0] if self._rows else None


class FakeSession:
    """Canned stand-in for AsyncSession that records what the endpoints do to it."""

    def __init__(self) -> None:
        """Start with nothing queued, stored, added, or committed."""
        self.statements: list[Any] = []
        self.added: list[Any] = []
        self.commits = 0
        self._queued: list[list[Any]] = []
        self._stored: dict[int, PromptVersion] = {}

    def queue(self, rows: list[Any]) -> None:
        """Queue the rows that the next unanswered execute() returns."""
        self._queued.append(rows)

    def store(self, row: PromptVersion) -> None:
        """Make get() return row for its id."""
        self._stored[row.id] = row

    async def execute(self, statement: Any) -> FakeResult:
        """Record the statement and return the next queued rows, or none for an UPDATE."""
        self.statements.append(statement)
        return FakeResult(self._queued.pop(0) if self._queued else [])

    async def get(self, entity: type, ident: int) -> PromptVersion | None:
        """Return the stored row, or None."""
        return self._stored.get(ident)

    def add(self, row: Any) -> None:
        """Record the new row."""
        self.added.append(row)

    async def flush(self) -> None:
        """Give every added row an id, as the database would."""
        for index, row in enumerate(self.added, start=100):
            row.id = index

    async def commit(self) -> None:
        """Count the commit."""
        self.commits += 1


def make_prompt(prompt_id: int, *, active: bool, content: str = "prompt text") -> PromptVersion:
    """Build a PromptVersion row as the database would return it."""
    row = PromptVersion(
        hash=f"hash{prompt_id:08d}", content=content, label=f"v{prompt_id}", is_active=active
    )
    row.id = prompt_id
    row.created_at = NOW
    return row


@pytest.fixture
def session() -> FakeSession:
    """The canned session that every endpoint reads and writes through."""
    return FakeSession()


@pytest.fixture
def prompt_file(monkeypatch, tmp_path) -> Path:
    """A temporary file that stands in for PROMPT.md."""
    path = tmp_path / "PROMPT.md"
    monkeypatch.setattr(prompts_routes, "PROMPT_PATH", path)
    return path


@pytest.fixture
def client(monkeypatch, session, prompt_file):
    """TestClient with admin auth bypassed and the canned session."""
    test_client = create_test_client(monkeypatch, Pipelines())
    from api.main import app

    async def no_admin_check() -> None:
        """Stand-in for require_admin_token."""

    async def canned_session() -> FakeSession:
        """Stand-in for require_db_session so no real database is needed."""
        return session

    monkeypatch.setitem(app.dependency_overrides, require_admin_token, no_admin_check)
    monkeypatch.setitem(app.dependency_overrides, require_db_session, canned_session)
    with test_client:
        yield test_client


def test_list_prompts_omits_content(client, session):
    """The list shows each version's metadata but never its possibly large text."""
    session.queue([make_prompt(2, active=True), make_prompt(1, active=False)])

    response = client.get("/prompts")

    assert response.status_code == 200
    body = response.json()
    assert [p["id"] for p in body] == [2, 1]
    assert body[0]["is_active"] is True
    assert "content" not in body[0]
    assert "created_at" in body[0]


def test_get_active_prompt_returns_content(client, session):
    """The active endpoint includes the full prompt text."""
    session.queue([make_prompt(2, active=True, content="be concise")])

    response = client.get("/prompts/active")

    assert response.status_code == 200
    assert response.json()["content"] == "be concise"
    assert response.json()["id"] == 2


def test_get_active_prompt_404_when_none_active(client, session):
    """With no active version the endpoint answers 404 rather than an empty body."""
    session.queue([])

    assert client.get("/prompts/active").status_code == 404


def test_create_prompt_deactivates_previous_and_writes_file(client, session, prompt_file):
    """A new version is hashed, stored as the only active one, and written to PROMPT.md."""
    session.queue([])  # no stored row has this hash
    response = client.post("/prompts", json={"content": "new prompt", "label": "tighter"})

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == 100
    assert body["hash"] == prompts_routes.compute_prompt_hash("new prompt")
    assert len(body["hash"]) == 64
    # The first statement clears is_active on all existing rows before the insert.
    assert "UPDATE geoai.prompt_versions" in str(session.statements[1])
    created = session.added[0]
    assert created.is_active is True
    assert created.label == "tighter"
    assert session.commits == 1
    assert prompt_file.read_text(encoding="utf-8") == "new prompt"


def test_create_prompt_survives_unwritable_prompt_file(client, session, monkeypatch, tmp_path):
    """A failed PROMPT.md write is only logged; the version is still saved."""
    monkeypatch.setattr(prompts_routes, "PROMPT_PATH", tmp_path / "missing_dir" / "PROMPT.md")

    response = client.post("/prompts", json={"content": "new prompt"})

    assert response.status_code == 201
    assert session.commits == 1


def test_create_prompt_reuses_row_with_same_hash(client, session, prompt_file):
    """Saving text that is already stored reactivates its row instead of adding a duplicate."""
    existing = make_prompt(7, active=False, content="same text")
    existing.hash = prompts_routes.compute_prompt_hash("same text")
    session.queue([existing])

    response = client.post("/prompts", json={"content": "same text", "label": "renamed"})

    assert response.status_code == 200
    assert response.json() == {"id": 7, "hash": existing.hash}
    assert session.added == []
    assert existing.is_active is True
    assert existing.label == "renamed"
    assert session.commits == 1
    assert prompt_file.read_text(encoding="utf-8") == "same text"


def test_create_prompt_rejects_empty_content(client):
    """An empty prompt would blank the pipeline's instructions, so it is a validation error."""
    assert client.post("/prompts", json={"content": ""}).status_code == 422


def test_activate_prompt_switches_active_version(client, session, prompt_file):
    """Activating an old version makes it active and restores its text in PROMPT.md."""
    old = make_prompt(1, active=False, content="old text")
    session.store(old)

    response = client.post("/prompts/1/activate")

    assert response.status_code == 200
    assert response.json() == {"id": 1, "hash": old.hash}
    assert old.is_active is True
    assert "UPDATE geoai.prompt_versions" in str(session.statements[0])
    assert session.commits == 1
    assert prompt_file.read_text(encoding="utf-8") == "old text"


def test_activate_unknown_prompt_404(client, session):
    """An id with no row answers 404 and changes nothing."""
    response = client.post("/prompts/999/activate")

    assert response.status_code == 404
    assert session.statements == []
    assert session.commits == 0


class FakeFactory:
    """Stand-in for async_sessionmaker: calling it gives an async context manager of the session."""

    def __init__(self, session: FakeSession) -> None:
        """Hold the session to hand out."""
        self._session = session

    def __call__(self) -> FakeFactory:
        """Return self as the context manager."""
        return self

    async def __aenter__(self) -> FakeSession:
        """Enter with the session."""
        return self._session

    async def __aexit__(self, *exc: object) -> None:
        """Leave without suppressing errors."""


def test_seed_inserts_prompt_file_when_table_empty(session, tmp_path):
    """The first startup stores PROMPT.md as the active version."""
    path = tmp_path / "PROMPT.md"
    path.write_text("seed text", encoding="utf-8")
    session.queue([])

    asyncio.run(_seed_system_prompt(FakeFactory(session), path))  # type: ignore[arg-type]

    assert len(session.added) == 1
    assert session.added[0].content == "seed text"
    assert session.added[0].is_active is True
    assert session.commits == 1


def test_seed_skips_when_prompts_exist(session, tmp_path):
    """Later startups leave the stored versions alone when one is already active."""
    path = tmp_path / "PROMPT.md"
    path.write_text("seed text", encoding="utf-8")
    session.queue([1])

    asyncio.run(_seed_system_prompt(FakeFactory(session), path))  # type: ignore[arg-type]

    assert session.added == []
    assert session.commits == 0


def test_seed_activates_existing_row_with_same_hash(session, tmp_path):
    """A row stored earlier by the deployment bootstrap is activated, not inserted twice."""
    path = tmp_path / "PROMPT.md"
    path.write_text("seed text", encoding="utf-8")
    existing = make_prompt(3, active=False, content="seed text")
    session.queue([])  # nothing is active yet
    session.queue([existing])  # but this text is already stored

    asyncio.run(_seed_system_prompt(FakeFactory(session), path))  # type: ignore[arg-type]

    assert session.added == []
    assert existing.is_active is True
    assert session.commits == 1
