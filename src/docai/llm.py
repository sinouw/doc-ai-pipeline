"""Minimal client for Groq's OpenAI-compatible chat API, with rate-limit aware retries."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .schema import Usage

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "qwen/qwen3.8-27b"


class LlmError(RuntimeError):
    pass


class QuotaExhaustedError(LlmError):
    """A daily quota ran out: retrying within minutes cannot succeed."""


@dataclass
class ChatResult:
    content: str
    usage: Usage


class GroqClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 12,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise LlmError("GROQ_API_KEY is not set")
        self.model = model or os.environ.get("DOCAI_MODEL", DEFAULT_MODEL)
        self.max_retries = max_retries
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
            transport=transport,
        )

    def chat(self, messages: list[dict[str, Any]], *, max_tokens: int = 1500, json_mode: bool = True) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        retries = 0
        waited = 0.0
        while True:
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                if retries >= self.max_retries:
                    raise LlmError(f"transport error: {exc}") from exc
                retries += 1
                waited += min(2**retries, 30)
                time.sleep(min(2**retries, 30))
                continue
            if response.status_code == 429 and "per day" in response.text:
                raise QuotaExhaustedError(f"daily quota reached: {response.text[:300]}")
            if response.status_code == 429 or response.status_code >= 500:
                if retries >= self.max_retries:
                    raise LlmError(f"HTTP {response.status_code} after {retries} retries: {response.text[:300]}")
                retries += 1
                # The limit is shared per organisation, so retry-after alone can be too optimistic.
                wait = max(_retry_after(response) or 0.0, min(2.0**retries, 60.0))
                log.info("groq %s, retry %d in %.1fs", response.status_code, retries, wait)
                waited += wait
                time.sleep(wait)
                continue
            if response.status_code != 200:
                raise LlmError(f"HTTP {response.status_code}: {response.text[:500]}")
            data = response.json()
            usage = data.get("usage") or {}
            return ChatResult(
                content=data["choices"][0]["message"].get("content") or "",
                usage=Usage(
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    calls=1,
                    retries=retries,
                    wait_ms=waited * 1000,
                ),
            )

    def close(self) -> None:
        self._http.close()


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return float(value) + 0.5 if value else None
    except ValueError:
        return None


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating code fences and leading reasoning text."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{") :]
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < start:
        raise LlmError(f"no JSON object in model output: {text[:200]!r}")
    try:
        obj = json.loads(s[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LlmError(f"invalid JSON from model: {exc}") from exc
    if not isinstance(obj, dict):
        raise LlmError("model output is not a JSON object")
    return obj
