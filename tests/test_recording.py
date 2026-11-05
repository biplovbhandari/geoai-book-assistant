"""Unit tests for the pure helper functions in api.recording.

record_interaction itself opens a real PostgreSQL session and is out of
scope here; only the database-free row-shaping and label-derivation logic
is covered.
"""

import uuid

from api.recording import (
    RecordingContext,
    _build_chunk_and_citation_rows,
    _build_trace_rows,
    _derive_chunk_display,
)


class TestDeriveChunkDisplay:
    def test_chapter_and_section(self):
        """Both chapter and section produces 'chapter - section' label."""
        label, url = _derive_chunk_display(
            "book_text",
            {"source_path": "book/03/index.qmd", "chapter": "03", "section": "U-Net"},
        )
        assert label == "03 - U-Net"
        assert url == "/03/index.html"

    def test_chapter_only(self):
        label, _url = _derive_chunk_display(
            "book_text", {"source_path": "book/03/index.qmd", "chapter": "03"}
        )
        assert label == "03"

    def test_section_only(self):
        label, _url = _derive_chunk_display(
            "book_text", {"source_path": "book/03/index.qmd", "section": "U-Net"}
        )
        assert label == "U-Net"

    def test_no_chapter_no_section_uses_filename(self):
        label, _url = _derive_chunk_display("book_text", {"source_path": "book/03/index.qmd"})
        assert label == "index"

    def test_video_transcript_uses_watch_link(self):
        label, url = _derive_chunk_display(
            "video_transcript",
            {
                "source_path": "data/transcripts/vid1.json",
                "chapter": "01",
                "section": "Intro",
                "watch_link": "https://youtu.be/abc123",
            },
        )
        assert url == "https://youtu.be/abc123"
        assert label == "01 - Intro"

    def test_non_book_path_returns_none_url(self):
        _label, url = _derive_chunk_display("book_text", {"source_path": "data/other.txt"})
        assert url is None

    def test_missing_source_path_uses_unknown(self):
        label, _url = _derive_chunk_display("book_text", {})
        assert label == "unknown"

    def test_none_source_path_uses_unknown(self):
        """Explicit None in metadata should fall back to 'unknown', not 'None'."""
        label, _url = _derive_chunk_display("book_text", {"source_path": None})
        assert label == "unknown"


class TestBuildChunkAndCitationRows:
    def test_basic_row_construction(self):
        scored_chunks = [
            {
                "chunk_id": "ch-001",
                "content": "test content",
                "source_type": "book_text",
                "metadata": {
                    "source_path": "book/03/index.qmd",
                    "chapter": "03",
                    "section": "U-Net",
                },
                "score": 0.95,
                "ranking_method": "dense",
            }
        ]
        interaction_id = uuid.uuid4()
        chunk_rows, citation_rows = _build_chunk_and_citation_rows(
            scored_chunks, interaction_id, index_run_id=1
        )

        assert len(chunk_rows) == 1
        assert chunk_rows[0]["chunk_id"] == "ch-001"
        assert chunk_rows[0]["display_label"] == "03 - U-Net"
        assert chunk_rows[0]["source_path"] == "book/03/index.qmd"
        assert len(citation_rows) == 1
        assert citation_rows[0]["citation_index"] == 0
        assert citation_rows[0]["score"] == 0.95

    def test_empty_chunks(self):
        chunk_rows, citation_rows = _build_chunk_and_citation_rows([], uuid.uuid4(), index_run_id=1)
        assert chunk_rows == []
        assert citation_rows == []


class TestBuildTraceRows:
    def test_produces_three_trace_rows(self):
        ctx = RecordingContext(
            interaction_id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            visitor_id=uuid.uuid4(),
            is_new_conversation=True,
            ga4_client_id=None,
            question="Q?",
            model_name="test-model",
            retrieval_ms=42,
            generation_wall_ms=100,
            token_event_count=5,
            scored_chunks=[{"chunk_id": "ch-001"}],
            retrieval_query="Q?",
        )
        traces = _build_trace_rows(ctx)

        assert len(traces) == 3
        assert traces[0]["stage"] == "retrieval"
        assert traces[0]["latency_ms"] == 42
        assert traces[1]["stage"] == "context"
        assert traces[1]["data"] == {}, "no messages reached the LLM, so context data is empty"
        assert traces[2]["stage"] == "generation"
        assert traces[2]["latency_ms"] == 100
        assert traces[2]["data"]["model"] == "test-model"


class TestRecordingContext:
    def test_defaults(self):
        ctx = RecordingContext(
            interaction_id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            visitor_id=uuid.uuid4(),
            is_new_conversation=True,
            ga4_client_id=None,
            question="Q?",
            model_name="m",
        )

        assert ctx.response_text == ""
        assert ctx.token_event_count == 0
        assert ctx.saw_done is False
        assert ctx.scored_chunks == []
