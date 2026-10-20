from __future__ import annotations

from collections.abc import Iterator


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
    ) -> str:
        """Send a chat completion request and return the response text.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the generated response.

        Returns:
            The generated text content.

        Raises:
            RuntimeError: If the response has no choices or empty content.
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
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("LLM returned empty content")
        return content

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Stream a chat completion response, yielding text chunks.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the generated response.

        Yields:
            Text chunks as they are produced by the model.
        """
        kwargs: dict = {
            "model": self._model,
            "messages": messages,  # type: ignore[arg-type]
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **self._extra_create_params,
        }
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        response = self._client.chat.completions.create(**kwargs)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
