from __future__ import annotations

from pathlib import Path

from geoai_rag.models.scored_chunk import ScoredChunk

_PROMPT_PATH = Path(__file__).parent / "PROMPT.md"
SYSTEM_PROMPT = _PROMPT_PATH.read_text().strip()

_HISTORY_BUDGET_PCT = 0.60
_CHARS_PER_TOKEN = 4
_FALLBACK_HISTORY_TOKENS = 1000


class DefaultContextBuilder:
    """Assembles system + user messages from retrieved chunks."""

    def __init__(self, max_history_tokens: int = _FALLBACK_HISTORY_TOKENS) -> None:
        """Initialize with a history token budget.

        Args:
            max_history_tokens: Maximum tokens of conversation history to include.
                Derived from ``LLM_CONTEXT_WINDOW * 0.60`` at startup.
                Internally converted to character caps.
        """
        max_chars = max_history_tokens * _CHARS_PER_TOKEN
        self._max_history_chars = max_chars
        self._per_message_cap = max(max_chars // 4, 200)
        self._pre_escape_cap = self._per_message_cap * 2

    @staticmethod
    def _xml_escape(text: str) -> str:
        """Escape XML-special characters to prevent tag injection."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _format_history(self, history: list[dict[str, str]]) -> str:
        """Format history as XML-delimited turns, capped by the token budget.

        Content is XML-escaped to prevent injection of closing tags.
        Escaping happens before truncation so the cap applies to final output size.
        Per-message and total caps scale with the context window via the
        constructor's ``max_history_tokens``.
        """
        turns = []
        for msg in history:
            role = msg.get("role", "user")
            if role not in ("user", "assistant"):
                role = "user"
            content = str(msg.get("content", ""))
            if len(content) > self._pre_escape_cap:
                content = content[: self._pre_escape_cap]
            content = DefaultContextBuilder._xml_escape(content)
            if len(content) > self._per_message_cap:
                content = content[: self._per_message_cap] + "..."
            turns.append(f'<turn role="{role}">{content}</turn>')

        result = "\n".join(turns)
        while len(result) > self._max_history_chars and turns:
            turns.pop(0)
            result = "\n".join(turns)
        return result

    def build(
        self,
        question: str,
        chunks: list[ScoredChunk],
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Assemble system and user messages from retrieved chunks.

        Args:
            question: The user's natural language question.
            chunks: Ranked chunks from retrieval.
            history: Optional prior conversation turns for reference resolution.

        Returns:
            List of message dicts (system + user) ready for LLMClient.chat().
        """
        context_parts = []
        for i, scored in enumerate(chunks, 1):
            source = scored.chunk.metadata.get("source_path", "unknown")
            section = scored.chunk.metadata.get("section", "")
            header = f"[{i}] {source}"
            if section:
                header += f" - {section}"
            context_parts.append(f"{header}\n{scored.chunk.content}")

        context = "\n\n---\n\n".join(context_parts)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            history_block = self._format_history(history)
            user_content = (
                f"<history>\n{history_block}\n</history>\n\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
        else:
            user_content = f"Context:\n{context}\n\nQuestion: {question}"

        messages.append({"role": "user", "content": user_content})
        return messages
