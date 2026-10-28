"""LLM extractors: OCR text only, page image only (vision), or both (hybrid)."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Literal

from .imaging import Gray, to_jpeg_data_url
from .llm import GroqClient, parse_json_object
from .parsing import decimals_for, normalize_currency, normalize_id, parse_amount, parse_date
from .schema import Invoice, LineItem, Usage

LlmMethod = Literal["ocr_llm", "vlm", "hybrid"]
PLAIN_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

SYSTEM_PROMPT = """You extract structured data from scanned invoices written in French, English or Arabic.
Return one JSON object with exactly these keys:
{
  "invoice_number": string|null,   // copy exactly as printed
  "issue_date": "YYYY-MM-DD"|null,
  "due_date": "YYYY-MM-DD"|null,
  "seller_name": string|null,      // the company issuing the invoice, as printed (keep Arabic script)
  "seller_vat_id": string|null,    // VAT number / matricule fiscal / ICE, as printed without spaces
  "buyer_name": string|null,       // the customer, as printed
  "currency": string|null,         // ISO 4217 code: EUR, GBP, TND, MAD...
  "subtotal": string|null,         // total before tax
  "vat_rate": string|null,         // percent, e.g. "20"
  "vat_amount": string|null,
  "total": string|null,            // total including tax, as printed even if it does not add up
  "line_items": [{"description": string, "quantity": string, "unit_price": string, "amount": string}]
}
Write every amount as a plain decimal number with a dot and the number of decimals printed (e.g. "1234.560"
for 1 234,560 TND). Dates on these invoices are day-first when numeric. Arabic-Indic digits must be
converted to 0-9. Use null for anything not printed. Never compute or correct values: report what is printed."""

OCR_HINT = (
    "OCR transcript of the page (rows in reading order, cells separated by 3 spaces). It may contain "
    "recognition errors; when the image is provided, the image wins."
)


def build_messages(method: LlmMethod, images: list[Gray], ocr_text: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if method in ("ocr_llm", "hybrid"):
        content.append({"type": "text", "text": f"{OCR_HINT}\n<ocr>\n{ocr_text}\n</ocr>"})
    if method in ("vlm", "hybrid"):
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": to_jpeg_data_url(image)}})
    content.append({"type": "text", "text": "Extract the invoice as JSON."})
    if method == "ocr_llm":
        user: dict[str, Any] = {"role": "user", "content": "\n\n".join(str(c["text"]) for c in content)}
    else:
        user = {"role": "user", "content": content}
    return [{"role": "system", "content": SYSTEM_PROMPT}, user]


def coerce_invoice(raw: dict[str, Any]) -> Invoice:
    """Turn loosely typed model output into an Invoice, parsing locale formats instead of trusting them."""
    currency = normalize_currency(str(raw.get("currency") or "")) or (raw.get("currency") or None)
    decimals = decimals_for(currency if isinstance(currency, str) else None)

    def amount(value: Any) -> Decimal | None:
        if value is None or value == "":
            return None
        if isinstance(value, int | float):
            return Decimal(str(value))
        s = str(value).strip()
        if PLAIN_NUMBER.fullmatch(s):  # the format we asked for
            return Decimal(s)
        return parse_amount(s, decimals)

    def text(value: Any) -> str | None:
        return str(value).strip() or None if value not in (None, "") else None

    items = []
    for li in raw.get("line_items") or []:
        if not isinstance(li, dict):
            continue
        qty, unit, amt = amount(li.get("quantity")), amount(li.get("unit_price")), amount(li.get("amount"))
        if qty is None or unit is None or amt is None:
            continue
        items.append(LineItem(description=str(li.get("description") or "").strip(), quantity=qty, unit_price=unit, amount=amt))
    rate = amount(raw.get("vat_rate"))
    return Invoice(
        invoice_number=text(raw.get("invoice_number")),
        issue_date=parse_date(text(raw.get("issue_date"))),
        due_date=parse_date(text(raw.get("due_date"))),
        seller_name=text(raw.get("seller_name")),
        seller_vat_id=normalize_id(text(raw.get("seller_vat_id"))),
        buyer_name=text(raw.get("buyer_name")),
        currency=currency if isinstance(currency, str) else None,
        subtotal=amount(raw.get("subtotal")),
        vat_rate=_plain(rate),
        vat_amount=amount(raw.get("vat_amount")),
        total=amount(raw.get("total")),
        line_items=items,
    )


def _plain(value: Decimal | None) -> Decimal | None:
    """20.00 -> 20, 5.50 -> 5.5, never scientific notation."""
    if value is None:
        return None
    return value.quantize(Decimal(1)) if value == value.to_integral_value() else value.normalize()


def extract_llm(client: GroqClient, method: LlmMethod, images: list[Gray], ocr_text: str) -> tuple[Invoice, Usage]:
    usage = Usage()
    messages = build_messages(method, images, ocr_text)
    result = client.chat(messages)
    usage.add(result.usage)
    try:
        raw = parse_json_object(result.content)
    except Exception:
        # One repair attempt: show the model its own output and ask for valid JSON only.
        messages = [
            *messages,
            {"role": "assistant", "content": result.content},
            {"role": "user", "content": "That was not a valid JSON object. Reply with the JSON object only."},
        ]
        result = client.chat(messages)
        usage.add(result.usage)
        raw = parse_json_object(result.content)
    return coerce_invoice(raw), usage
