from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel


class GenerationResult(BaseModel):
    """Structured result from a synchronous LLM call."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    model: str = ""
    reasoning_content: str | None = None
    reasoning_tokens: int | None = None


class StreamingGenerationResult:
    """Wraps an LLM streaming response to capture metadata after iteration.

    Iterate for text chunks. After iteration completes, read ``.metadata``
    for token counts, finish_reason, etc. ``.metadata`` is None until the
    stream has been fully consumed, and its token counts are None if the
    server did not provide usage data.
    """

    def __init__(self, stream: Iterator) -> None:
        """Wrap a raw streaming response.

        Args:
            stream: Iterator of chunk objects from an OpenAI-compatible
                streaming chat completion.
        """
        self._stream = stream
        self.metadata: GenerationResult | None = None

    def __iter__(self) -> Iterator[str]:
        """Yield text deltas as they arrive, then populate ``metadata``.

        Usage, model name, and finish reason are read from whichever chunks
        carry them. The wrapped stream is consumed during iteration, so
        iterate only once.

        Yields:
            The non-empty text content of each streamed chunk.
        """
        finish_reason: str | None = None
        model = ""
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        for chunk in self._stream:
            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = getattr(chunk.usage, "prompt_tokens", None)
                completion_tokens = getattr(chunk.usage, "completion_tokens", None)

            if hasattr(chunk, "model") and chunk.model:
                model = chunk.model

            if hasattr(chunk, "choices") and chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            if (
                hasattr(chunk, "choices")
                and chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                yield chunk.choices[0].delta.content

        self.metadata = GenerationResult(
            text="",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            model=model,
        )
