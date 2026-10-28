from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from docai.extract import build_messages, coerce_invoice, extract_llm
from docai.llm import GroqClient, LlmError, parse_json_object

from .conftest import blank_page


def completion(content: str, prompt: int = 100, out: int = 50) -> httpx.Response:
    body = {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": prompt, "completion_tokens": out}}
    return httpx.Response(200, json=body)


def client_for(responses: list[httpx.Response], seen: list[dict[str, Any]] | None = None) -> GroqClient:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return queue.pop(0)

    return GroqClient(api_key="test", model="test-model", transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr("docai.llm.time.sleep", slept.append)
    return slept


def test_retries_rate_limits_using_retry_after(no_sleep: list[float]) -> None:
    client = client_for([httpx.Response(429, headers={"retry-after": "7"}), completion('{"a": 1}')])
    result = client.chat([{"role": "user", "content": "hi"}])
    assert result.usage.retries == 1
    assert result.usage.calls == 1
    assert no_sleep == [7.5]


def test_gives_up_after_max_retries() -> None:
    client = client_for([httpx.Response(503)] * 3)
    client.max_retries = 2
    with pytest.raises(LlmError, match="503"):
        client.chat([{"role": "user", "content": "hi"}])


def test_client_errors_are_not_retried() -> None:
    client = client_for([httpx.Response(400, json={"error": "bad"})])
    with pytest.raises(LlmError, match="400"):
        client.chat([{"role": "user", "content": "hi"}])


def test_parse_json_object_tolerates_fences() -> None:
    assert parse_json_object('```json\n{"total": "1.00"}\n```') == {"total": "1.00"}
    with pytest.raises(LlmError):
        parse_json_object("no json here")


def test_coerce_invoice_parses_locale_formats() -> None:
    inv = coerce_invoice(
        {
            "invoice_number": " FA-1 ",
            "issue_date": "14/03/2026",
            "seller_vat_id": "FR 40 303 265 045",
            "currency": "€",
            "subtotal": "1 234,56",
            "vat_rate": "20.0",
            "total": 1481.47,
            "line_items": [{"description": "x", "quantity": "1", "unit_price": "1234.56", "amount": "1234.56"}, {"bad": 1}],
        }
    )
    assert inv.invoice_number == "FA-1"
    assert str(inv.issue_date) == "2026-03-14"
    assert inv.seller_vat_id == "FR40303265045"
    assert inv.currency == "EUR"
    assert inv.subtotal == Decimal("1234.56")
    assert str(inv.vat_rate) == "20"
    assert inv.total == Decimal("1481.47")
    assert len(inv.line_items) == 1


def test_coerce_keeps_three_decimals_for_dinar() -> None:
    assert coerce_invoice({"currency": "TND", "total": "343.624"}).total == Decimal("343.624")
    assert coerce_invoice({"currency": "TND", "total": "343,624"}).total == Decimal("343.624")


def test_messages_per_method() -> None:
    page = blank_page()
    ocr_only = build_messages("ocr_llm", [page], "OCR TEXT")
    assert isinstance(ocr_only[1]["content"], str) and "OCR TEXT" in ocr_only[1]["content"]
    vlm = build_messages("vlm", [page], "OCR TEXT")
    assert [c["type"] for c in vlm[1]["content"]] == ["image_url", "text"]
    hybrid = build_messages("hybrid", [page], "OCR TEXT")
    assert [c["type"] for c in hybrid[1]["content"]] == ["text", "image_url", "text"]


def test_extract_llm_repairs_invalid_json_once() -> None:
    seen: list[dict[str, Any]] = []
    client = client_for([completion("sorry, here it is: total 5"), completion('{"total": "5.00", "currency": "EUR"}')], seen)
    inv, usage = extract_llm(client, "vlm", [blank_page()], "")
    assert inv.total == Decimal("5.00")
    assert usage.calls == 2
    assert seen[1]["messages"][-1]["role"] == "user"
    assert seen[0]["response_format"] == {"type": "json_object"}


def test_parse_duration() -> None:
    from docai.llm import parse_duration

    assert parse_duration("16.717s") == 16.717
    assert parse_duration("1m26.4s") == 86.4
    assert parse_duration("11h9m36s") == 11 * 3600 + 9 * 60 + 36
    assert parse_duration("1ms") == 0.001
    assert parse_duration(None) is None


def test_paces_on_remaining_tokens(no_sleep: list[float]) -> None:
    headers = {"x-ratelimit-remaining-tokens": "500", "x-ratelimit-reset-tokens": "20s"}
    first = httpx.Response(200, headers=headers, json={"choices": [{"message": {"content": "{}"}}], "usage": {}})
    client = client_for([first, completion("{}")])
    client.chat([{"role": "user", "content": "one"}])
    assert no_sleep == []
    result = client.chat([{"role": "user", "content": "two"}])  # needs ~700 tokens, only 500 left
    assert len(no_sleep) == 1 and 19 < no_sleep[0] <= 20
    assert result.usage.wait_ms > 0


def test_cache_returns_stored_response_without_calling(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    first = client_for([completion('{"a": 1}', prompt=1900, out=300)], calls)
    first.cache_dir = tmp_path
    messages = [{"role": "user", "content": "same"}]
    assert first.chat(messages).content == '{"a": 1}'
    again = client_for([], calls)
    again.cache_dir = tmp_path
    cached = again.chat(messages)
    assert cached.content == '{"a": 1}'
    assert cached.usage.prompt_tokens == 1900  # original cost kept for accounting
    assert len(calls) == 1
