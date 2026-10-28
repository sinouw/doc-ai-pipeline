"""Per-field confidence from independent signals.

* validation  - the field takes part in a passing arithmetic / checksum check (1) or only failing ones (0)
* ocr_support - the value can be found in the OCR output, weighted by the OCR line score
* agreement   - an independent extractor (the OCR rules) produced the same value

The confidence is the weighted mean of the signals that exist for that field. Missing values score 0.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from .compare import AMOUNT_FIELDS, DATE_FIELDS, ID_FIELDS, NAME_FIELDS, values_equal
from .ocr import OcrPage
from .parsing import decimals_for, find_amounts, find_dates, normalize_id, normalize_text
from .schema import REVIEW_FIELDS, FieldResult, FieldStatus, Invoice, LineItem
from .validate import Check, field_validation_scores

WEIGHTS = {"validation": 0.4, "ocr_support": 0.3, "agreement": 0.3}
DEFAULT_THRESHOLD = 0.8


class OcrIndex:
    """Pre-normalised views of the OCR output for fast lookups."""

    def __init__(self, pages: list[OcrPage], currency: str | None) -> None:
        self.lines = [ln for p in pages for ln in p.lines]
        self.texts = [normalize_text(ln.text) for ln in self.lines]
        self.full = "\n".join(p.text() for p in pages)
        self.compact = normalize_id(self.full.replace("\n", " ")) or ""
        self.dates = set(find_dates(self.full))
        decimals = decimals_for(currency)
        self.amounts: dict[Any, float] = {}
        for ln in self.lines:
            for value in find_amounts(ln.text, decimals):
                self.amounts[value] = max(self.amounts.get(value, 0.0), ln.score)
        self.mean_score = sum(ln.score for ln in self.lines) / len(self.lines) if self.lines else 0.0

    def support(self, field: str, value: Any) -> float:
        if value is None:
            return 0.0
        if field in AMOUNT_FIELDS:
            if field == "vat_rate":
                return (
                    1.0 if re.search(rf"(?<!\d){re.escape(str(value))}\s*%|%\s*{re.escape(str(value))}(?!\d)", self.full) else 0.0
                )
            return self.amounts.get(value, 0.0)
        if field in DATE_FIELDS:
            return self.mean_score if value in self.dates else 0.0
        if field in ID_FIELDS:
            key = normalize_id(str(value)) or ""
            return self.mean_score if key and key in self.compact else 0.0
        if field in NAME_FIELDS:
            target = normalize_text(str(value))
            if not target:
                return 0.0
            best = 0.0
            for text, line in zip(self.texts, self.lines, strict=True):
                ratio = 1.0 if target in text else SequenceMatcher(None, target, text).ratio()
                best = max(best, ratio * line.score)
            return best if best >= 0.75 else 0.0
        if field == "currency":
            return 1.0 if str(value) in self.full else 0.5
        if field == "line_items":
            items: list[LineItem] = value
            if not items:
                return 0.0
            found = sum(1 for li in items if li.amount in self.amounts and li.unit_price in self.amounts)
            return found / len(items)
        return 0.0


def needs_review(confidence: float, signals: dict[str, float], threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Below the threshold, or blamed by validation (in failing checks and in no passing one)."""
    return confidence < threshold or signals.get("validation") == 0.0


def score_fields(
    invoice: Invoice,
    checks: list[Check],
    pages: list[OcrPage],
    reference: Invoice | None,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[FieldResult]:
    index = OcrIndex(pages, invoice.currency)
    validation = field_validation_scores(checks)
    results = []
    for name in REVIEW_FIELDS:
        value = getattr(invoice, name)
        empty = value is None or value == []
        signals: dict[str, float] = {}
        if name in validation:
            signals["validation"] = validation[name]
        signals["ocr_support"] = round(index.support(name, value), 4)
        if reference is not None:
            ref_value = getattr(reference, name)
            ref_empty = ref_value is None or ref_value == []
            signals["agreement"] = 0.5 if ref_empty else float(values_equal(name, value, ref_value))
        issues = [c.message or c.code for c in checks if c.outcome.value == "fail" and name in c.fields]
        if empty:
            confidence = 0.0
            issues.append("missing")
        else:
            total_w = sum(WEIGHTS[k] for k in signals)
            confidence = sum(WEIGHTS[k] * v for k, v in signals.items()) / total_w
        status = FieldStatus.NEEDS_REVIEW if needs_review(confidence, signals, threshold) else FieldStatus.AUTO_ACCEPTED
        results.append(
            FieldResult(
                name=name,
                value=_jsonable(value),
                confidence=round(confidence, 4),
                signals=signals,
                issues=issues,
                status=status,
            )
        )
    return results


def _jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return [li.model_dump(mode="json") if isinstance(li, LineItem) else li for li in value]
    if value is None or isinstance(value, str):
        return value
    return str(value)
