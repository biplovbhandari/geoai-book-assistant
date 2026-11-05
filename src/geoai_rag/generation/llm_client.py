from __future__ import annotations

import re

from geoai_rag.models.generation_result import GenerationResult, StreamingGenerationResult

# Qwen3-style reasoning models emit their chain of thought as a <think>...</think> block
# ahead of the answer.
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _extract_reasoning(text: str) -> tuple[str, str | None]:
    """Separate reasoning content from answer text.

    Args:
        text: Raw model output, possibly containing a ``<think>...</think>`` block.

    Returns:
        A tuple of (answer_text, reasoning_content). Text without a think block is returned
        unchanged with None. Otherwise every think block is removed from the answer, which is
        stripped, and the reasoning is the stripped content of the first block (None if blank).
    """
    match = _THINK_PATTERN.search(text)
    if not match:
        return text, None
    reasoning = match.group(1).strip()
    answer = _THINK_PATTERN.sub("", text).strip()
    return answer, reasoning if reasoning else None


class OpenAICompatibleClient:
    """LLM client using the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        reasoning_effort: str = "none",
        extra_create_params: dict | None = None,
    ) -> None:
        """Initialize the client with connection parameters.

        Args:
            base_url: API endpoint URL (Ollama, NASA proxy, etc.).
            api_key: Authentication key for the API.
            model: Model name to use for completions.
            timeout: Request timeout in seconds.
            reasoning_effort: Model thinking depth ("none", "low", "medium", "high").
                "none" disables reasoning; other values are passed as-is to the API.
            extra_create_params: Additional keyword arguments merged into every
                chat.completions.create() call (e.g. top_p, frequency_penalty).
        """
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._extra_create_params = extra_create_params or {}

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """Send a chat completion request and return the answer with generation metadata.

        A ``<think>...</think>`` block in the reply (Qwen3-style reasoning) is split off into
        ``reasoning_content`` so ``text`` holds only the answer.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the generated response.

        Returns:
            The answer text with the finish reason, resolved model name, any reasoning
            content, and token counts (None when the server reports no usage).

        Raises:
            RuntimeError: If the response has no choices or no answer text.
        """
        kwargs: dict = {
            "model": self._model,
            "messages": messages,  # type: ignore[arg-type]
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self._extra_create_params,
        }
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        response = self._client.chat.completions.create(**kwargs)
        if not response.choices:
            raise RuntimeError("LLM returned no choices")
        choice = response.choices[0]
        answer, reasoning_content = _extract_reasoning(choice.message.content or "")
        # Checked after extraction: a reply that is only a think block has no answer to return.
        if not answer.strip():
            raise RuntimeError("LLM returned empty content")
        usage = response.usage
        return GenerationResult(
            text=answer,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason,
            model=response.model or self._model,
            reasoning_content=reasoning_content,
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> StreamingGenerationResult:
        """Start a streaming chat completion and return a wrapper around the response.

        The request is sent when this method is called, so connection and request errors
        (including a server that rejects ``stream_options``) raise here rather than on the
        first iteration.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the generated response.

        Returns:
            An iterable of text chunks as the model produces them. Once it has been fully
            iterated, its ``metadata`` attribute holds the finish reason, model name, and
            token counts (None when the server reports no usage).
        """
        kwargs: dict = {
            "model": self._model,
            "messages": messages,  # type: ignore[arg-type]
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self._extra_create_params,
        }
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        response = self._client.chat.completions.create(**kwargs)
        return StreamingGenerationResult(response)
