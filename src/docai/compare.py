"""Field equality used by both confidence scoring (do two extractors agree?) and evaluation
(does a prediction match the label?)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from .parsing import normalize_id, normalize_text, parse_amount, parse_date
from .schema import LineItem

ID_FIELDS = {"invoice_number", "seller_vat_id"}
NAME_FIELDS = {"seller_name", "buyer_name"}
DATE_FIELDS = {"issue_date", "due_date"}
AMOUNT_FIELDS = {"subtotal", "vat_rate", "vat_amount", "total"}


def _amount(value: Any) -> Decimal | None:
    if value is None:
        return None
    return parse_amount(value if isinstance(value, Decimal | int | float) else str(value), decimals=3)


def _date(value: Any) -> date | None:
    return value if isinstance(value, date) else parse_date(str(value)) if value else None


def _items(value: Any) -> list[LineItem]:
    return [li if isinstance(li, LineItem) else LineItem.model_validate(li) for li in value or []]


def line_items_equal(a: Any, b: Any, with_description: bool = False) -> bool:
    left, right = _items(a), _items(b)
    if len(left) != len(right):
        return False
    for x, y in zip(left, right, strict=True):
        if any(abs(getattr(x, k) - getattr(y, k)) > Decimal("0.0005") for k in ("quantity", "unit_price", "amount")):
            return False
        if with_description and normalize_text(x.description) != normalize_text(y.description):
            return False
    return True


def values_equal(field: str, a: Any, b: Any) -> bool:
    if field == "line_items":
        return line_items_equal(a, b)
    if a is None or b is None:
        return a is None and b is None
    if field in ID_FIELDS:
        return normalize_id(str(a)) == normalize_id(str(b))
    if field in NAME_FIELDS:
        return normalize_text(str(a)) == normalize_text(str(b))
    if field in DATE_FIELDS:
        return _date(a) == _date(b)
    if field in AMOUNT_FIELDS:
        x, y = _amount(a), _amount(b)
        return x is not None and y is not None and abs(x - y) <= Decimal("0.0005")
    if field == "currency":
        return str(a).upper() == str(b).upper()
    return bool(a == b)
