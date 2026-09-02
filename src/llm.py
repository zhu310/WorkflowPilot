from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import Settings


class DeepSeekAdapter:
    """Small OpenAI-compatible adapter; no retries or output repair."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            f"{self.settings.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"DeepSeek request failed: {error.reason}") from error

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"Unexpected DeepSeek response: {json.dumps(body, ensure_ascii=False)}") from error
