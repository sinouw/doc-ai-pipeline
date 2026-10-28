"""Minimal client for Groq's OpenAI-compatible chat API, with rate-limit aware retries."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
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


IMAGE_TOKENS = 1850  # measured: a page image costs ~1.8k prompt tokens on qwen3.8-27b, whatever its size
EXPECTED_OUTPUT_TOKENS = 700


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough upper estimate of a request's token cost, used to pace calls under a tokens-per-minute limit."""
    images, chars = 0, 0
    for message in messages:
        content = message["content"]
        parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for part in parts:
            if part.get("type") == "image_url":
                images += 1
            else:
                chars += len(str(part.get("text", "")))
    return images * IMAGE_TOKENS + chars // 3 + EXPECTED_OUTPUT_TOKENS


def parse_duration(value: str | None) -> float | None:
    """Groq reset headers look like '16.717s', '1m26.4s', '11h9m36s' or '1ms'."""
    if not value:
        return None
    if value.endswith("ms") and value[:-2].replace(".", "", 1).isdigit():
        return float(value[:-2]) / 1000
    total, found = 0.0, False
    for amount, unit in re.findall(r"([\d.]+)([hms])", value):
        total += float(amount) * {"h": 3600, "m": 60, "s": 1}[unit]
        found = True
    return total if found else None


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
        cache_dir: Path | None = None,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise LlmError("GROQ_API_KEY is not set")
        self.model = model or os.environ.get("DOCAI_MODEL", DEFAULT_MODEL)
        self.max_retries = max_retries
        cache = cache_dir or (Path(os.environ["DOCAI_LLM_CACHE"]) if os.environ.get("DOCAI_LLM_CACHE") else None)
        self.cache_dir = cache
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
        # Last rate-limit state reported by the API, used to wait just long enough before the next call.
        self._remaining_tokens: int | None = None
        self._tokens_reset_at = 0.0
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
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        retries = 0
        waited = self._pace(estimate_tokens(messages))
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
                self._update_limits(response)
                if retries >= self.max_retries:
                    raise LlmError(f"HTTP {response.status_code} after {retries} retries: {response.text[:300]}")
                retries += 1
                wait = _retry_after(response) or min(2.0**retries, 60.0)
                log.info("groq %s, retry %d in %.1fs", response.status_code, retries, wait)
                waited += wait
                time.sleep(wait)
                continue
            self._update_limits(response)
            if response.status_code != 200:
                raise LlmError(f"HTTP {response.status_code}: {response.text[:500]}")
            data = response.json()
            usage = data.get("usage") or {}
            result = ChatResult(
                content=data["choices"][0]["message"].get("content") or "",
                usage=Usage(
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    calls=1,
                    retries=retries,
                    wait_ms=waited * 1000,
                ),
            )
            self._cache_put(key, result)
            return result

    def _pace(self, needed: int) -> float:
        """Sleep until the per-minute token bucket should hold `needed` tokens. Returns seconds slept."""
        if self._remaining_tokens is None or self._remaining_tokens >= needed:
            return 0.0
        wait = self._tokens_reset_at - time.monotonic()
        if wait <= 0:
            return 0.0
        log.info("pacing: %d tokens left this minute, need ~%d, waiting %.1fs", self._remaining_tokens, needed, wait)
        time.sleep(wait)
        self._remaining_tokens = None
        return wait

    def _update_limits(self, response: httpx.Response) -> None:
        remaining = response.headers.get("x-ratelimit-remaining-tokens")
        reset = parse_duration(response.headers.get("x-ratelimit-reset-tokens"))
        if remaining is not None and remaining.isdigit():
            self._remaining_tokens = int(remaining)
            self._tokens_reset_at = time.monotonic() + (reset or 0.0)

    def _cache_get(self, key: str) -> ChatResult | None:
        if not self.cache_dir or not (self.cache_dir / f"{key}.json").is_file():
            return None
        data = json.loads((self.cache_dir / f"{key}.json").read_text(encoding="utf-8"))
        # The original token usage is kept so cost accounting stays honest; no time was spent waiting.
        return ChatResult(content=data["content"], usage=Usage.model_validate({**data["usage"], "wait_ms": 0.0}))

    def _cache_put(self, key: str, result: ChatResult) -> None:
        if self.cache_dir:
            payload = {"model": self.model, "content": result.content, "usage": result.usage.model_dump()}
            (self.cache_dir / f"{key}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

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
