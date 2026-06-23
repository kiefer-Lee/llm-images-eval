from __future__ import annotations

import os
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class VlmConfig:
    model: str
    api_key: str | None
    base_url: str | None
    temperature: float = 0.0
    max_tokens: int | None = None
    json_mode: str = "none"
    retries: int = 2
    retry_sleep: float = 2.0

    @classmethod
    def from_env(
        cls,
        model: str | None,
        api_key: str | None,
        base_url: str | None,
        temperature: float,
        max_tokens: int | None,
        json_mode: str,
        retries: int,
        retry_sleep: float,
    ) -> "VlmConfig":
        resolved_model = model or os.getenv("VLM_MODEL")
        if not resolved_model:
            raise ValueError("Missing model. Pass --model or set VLM_MODEL.")
        return cls(
            model=resolved_model,
            api_key=api_key or os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("VLM_BASE_URL"),
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            retries=retries,
            retry_sleep=retry_sleep,
        )


class OpenAICompatibleVlmClient:
    def __init__(self, config: VlmConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for API inference. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        kwargs = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)
        self.config = config

    def detect(self, prompt: str, image_data_url: str) -> str:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
        request = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            request["max_tokens"] = self.config.max_tokens
        if self.config.json_mode == "json_object":
            request["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                response = self.client.chat.completions.create(**request)
                if isinstance(response, str):
                    return response
                choice = response.choices[0]
                text = _message_to_text(choice.message)
                _raise_if_truncated(choice, text)
                return text
            except Exception as exc:  # noqa: BLE001 - API clients raise provider-specific exceptions.
                last_error = exc
                if attempt >= self.config.retries:
                    break
                time.sleep(self.config.retry_sleep * (attempt + 1))
        raise RuntimeError(f"VLM request failed after retries: {last_error}") from last_error


def _message_to_text(message: object) -> str:
    """Extract text from a chat message.

    Prefer ``content`` (the answer channel). Only fall back to
    ``reasoning_content`` when ``content`` is empty, since some reasoning
    models leave ``content`` blank. Note that a truncated reasoning response
    has no final answer to recover — :func:`_raise_if_truncated` guards that.
    """
    content = getattr(message, "content", None)
    text = _content_to_text(content) if content is not None else ""
    if text.strip():
        return text

    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    if reasoning:
        return _content_to_text(reasoning)
    return text


def _raise_if_truncated(choice: object, text: str) -> None:
    """Fail loudly when the model was cut off before emitting its answer.

    Reasoning models can consume the entire ``max_tokens`` budget while still
    thinking, returning ``finish_reason="length"`` with only partial reasoning
    and no JSON answer. Without this guard the downstream parser mines a stray
    bbox out of the chain-of-thought and silently reports zero detections.
    """
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        has_answer = bool(_content_to_text(content).strip()) if content is not None else False
        if not has_answer:
            raise RuntimeError(
                "Model response was truncated (finish_reason='length') before "
                "producing an answer. Increase --max-tokens; reasoning models "
                "need enough budget to think and then emit the JSON."
            )



def _content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
        return "\n".join(parts)
    return str(content)
