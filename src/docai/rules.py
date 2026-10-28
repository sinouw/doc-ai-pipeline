"""OCR-only baseline: keyword and pattern rules over OCR rows. No model calls.

Rows come from `group_rows` in reading order (right-to-left for Arabic), so on both scripts a value
follows its label. The rules know the label vocabulary in French, English and Arabic."""

from __future__ import annotations

import re
from decimal import Decimal

from .ocr import OcrLine, OcrPage
from .parsing import (
    decimals_for,
    find_dates,
    normalize_currency,
    normalize_digits,
    normalize_id,
    parse_amount,
)
from .schema import Invoice, LineItem

TITLE_WORDS = ("facture", "invoice", "فاتورة", "tax invoice", "facture proforma")
LABELS: dict[str, tuple[str, ...]] = {
    "invoice_number": ("facture n", "invoice no", "invoice number", "invoice #", "n° facture", "رقم الفاتورة"),
    "due_date": ("échéance", "echeance", "due date", "payment due", "تاريخ الاستحقاق"),
    "issue_date": ("invoice date", "date de facture", "date :", "date:", "date", "التاريخ", "تاريخ الفاتورة"),
    "buyer": ("client", "bill to", "billed to", "customer", "العميل", "الحريف"),
    "vat_id": ("tva", "vat", "المعرف الجبائي", "ice", "matricule fiscal"),
    "items_header": ("désignation", "description", "البيان"),
    "subtotal": ("total ht", "sous-total", "subtotal", "sub-total", "net amount", "المجموع دون"),
    "vat_amount": ("tva", "vat", "الأداء على القيمة", "ضريبة القيمة المضافة"),
    "total": ("total ttc", "total due", "amount due", "net à payer", "total", "المجموع الجملي", "المبلغ الإجمالي"),
    "currency": ("devise", "currency", "العملة"),
}
VAT_ID_PATTERNS = (
    re.compile(r"FR[0-9A-Z]{2}\d{9}"),
    re.compile(r"GB\d{9}(?:\d{3})?"),
    re.compile(r"IE\d{7}[A-W][A-I]?"),
    re.compile(r"(?:DE|ES|IT|BE|NL|LU|PT)[0-9A-Z]{8,12}"),
    re.compile(r"\d{7}[A-Z]/[A-Z]/[A-Z]/\d{3}"),  # Tunisian matricule fiscal
)
ICE_RE = re.compile(r"\b\d{15}\b")
NUMBER_TOKEN = re.compile(r"[A-Z]{0,5}[-/]?\d[\w\-/]*", re.IGNORECASE)


def _lower(text: str) -> str:
    return normalize_digits(text).lower()


def _has_label(text: str, field: str) -> bool:
    low = _lower(text)
    return any(label in low for label in LABELS[field])


def _row_text(row: list[OcrLine]) -> str:
    return "   ".join(c.text for c in row)


def _after_label(row: list[OcrLine], field: str) -> str:
    """Text of the row after the first cell carrying the label (including the cell's own tail)."""
    for i, cell in enumerate(row):
        low = _lower(cell.text)
        for label in LABELS[field]:
            pos = low.find(label)
            if pos >= 0:
                tail = cell.text[pos + len(label) :].lstrip(" :°.#Nn")
                return "   ".join([tail, *(c.text for c in row[i + 1 :])]).strip()
    return ""


def _amount_cells(cells: list[str], decimals: int) -> list[Decimal]:
    out = []
    for text in cells:
        if "%" in text or not re.search(r"\d", text):
            continue
        value = parse_amount(text, decimals)
        if value is not None:
            out.append(value)
    return out


def extract_rules(page: OcrPage) -> Invoice:
    rows = page.rows
    full_text = page.text()
    inv = Invoice()

    # Currency: explicit label first, then the most frequent code on the page.
    for row in rows:
        if _has_label(_row_text(row), "currency"):
            inv.currency = normalize_currency(_after_label(row, "currency"))
            if inv.currency:
                break
    if not inv.currency:
        codes = [c for c in (normalize_currency(t) for t in re.findall(r"[A-Z]{3}|[€£$]|د\.ت|DT|DH", full_text)) if c]
        inv.currency = max(set(codes), key=codes.count) if codes else None
    decimals = decimals_for(inv.currency)

    for row in rows:
        text = _row_text(row)
        if inv.invoice_number is None and _has_label(text, "invoice_number"):
            tail = _after_label(row, "invoice_number")
            m = NUMBER_TOKEN.search(normalize_digits(tail))
            if m:
                inv.invoice_number = m.group(0).strip("-/")
        if inv.due_date is None and _has_label(text, "due_date"):
            dates = find_dates(_after_label(row, "due_date"))
            inv.due_date = dates[0] if dates else None
        elif inv.issue_date is None and _has_label(text, "issue_date") and not _has_label(text, "due_date"):
            dates = find_dates(_after_label(row, "issue_date"))
            inv.issue_date = dates[0] if dates else None

    # VAT / tax id: the strongest pattern anywhere on the page, spaces removed.
    compact = normalize_id(full_text.replace("\n", " ")) or ""
    for pattern in VAT_ID_PATTERNS:
        m = pattern.search(compact)
        if m:
            inv.seller_vat_id = m.group(0)
            break
    if inv.seller_vat_id is None:
        m = ICE_RE.search(normalize_digits(full_text))
        inv.seller_vat_id = m.group(0) if m else None

    # Seller: top-most line that is not the document title.
    for line in sorted(page.lines, key=lambda ln: ln.box[1]):
        if not any(w in line.text.lower() for w in TITLE_WORDS) and len(line.text.strip()) > 2:
            inv.seller_name = line.text.strip()
            break

    # Buyer: the line right under the "Client" / "Bill to" label, overlapping horizontally.
    label_line = next((ln for ln in page.lines if _has_label(ln.text, "buyer")), None)
    if label_line is not None:
        below = [
            ln
            for ln in page.lines
            if ln.box[1] > label_line.cy and ln.box[0] < label_line.box[2] + 50 and ln.box[2] > label_line.box[0] - 50
        ]
        if below:
            inv.buyer_name = min(below, key=lambda ln: ln.box[1]).text.strip()

    # Totals block: specific labels before the generic "total".
    for row in rows:
        text = _row_text(row)
        cells = [c.text for c in row]
        amounts = _amount_cells(cells, decimals)
        if not amounts:
            continue
        if inv.subtotal is None and _has_label(text, "subtotal"):
            inv.subtotal = amounts[-1]
        elif inv.vat_amount is None and _has_label(text, "vat_amount") and "%" in text:
            rate = re.search(r"(\d+(?:[.,]\d+)?)\s*%|%\s*(\d+(?:[.,]\d+)?)", normalize_digits(text))
            if rate:
                inv.vat_rate = parse_amount(rate.group(1) or rate.group(2))
            inv.vat_amount = amounts[-1]
        elif inv.total is None and _has_label(text, "total"):
            inv.total = amounts[-1]

    inv.line_items = _line_items(rows, decimals)
    return inv


def _line_items(rows: list[list[OcrLine]], decimals: int) -> list[LineItem]:
    start = next((i for i, r in enumerate(rows) if _has_label(_row_text(r), "items_header")), None)
    if start is None:
        return []
    items = []
    for row in rows[start + 1 :]:
        text = _row_text(row)
        if _has_label(text, "subtotal") or _has_label(text, "total"):
            break
        numeric = [c.text for c in row if re.fullmatch(r"[\d\s.,٠-٩٫٬]+", c.text.strip())]
        words = [c.text for c in row if c.text not in numeric]
        values = _amount_cells(numeric, decimals)
        if len(values) >= 3:
            qty, unit, amount = values[-3:]
            items.append(LineItem(description=" ".join(words).strip(), quantity=qty, unit_price=unit, amount=amount))
    return items
