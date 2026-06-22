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
    max_tokens: int = 4096
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
        max_tokens: int,
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
            "max_tokens": self.config.max_tokens,
        }
        if self.config.json_mode == "json_object":
            request["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                response = self.client.chat.completions.create(**request)
                message = response.choices[0].message
                return _content_to_text(message.content)
            except Exception as exc:  # noqa: BLE001 - API clients raise provider-specific exceptions.
                last_error = exc
                if attempt >= self.config.retries:
                    break
                time.sleep(self.config.retry_sleep * (attempt + 1))
        raise RuntimeError(f"VLM request failed after retries: {last_error}") from last_error


def _content_to_text(content: object) -> str:
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
