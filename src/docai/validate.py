"""Field-level and cross-field validation.

Each check returns pass / fail / not applicable and names the fields it involves. A field that takes
part in at least one passing arithmetic check is *corroborated*: an OCR or model error rarely produces
numbers that still add up."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from .parsing import KNOWN_CURRENCIES, decimals_for
from .schema import Invoice, ValidationIssue


class Outcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NA = "n/a"


@dataclass(frozen=True)
class Check:
    code: str
    outcome: Outcome
    fields: tuple[str, ...]
    message: str = ""


def _tolerance(inv: Invoice, terms: int = 1) -> Decimal:
    """Rounding slack: half a unit of the last printed decimal per summed term, at least one unit."""
    unit = Decimal(1).scaleb(-decimals_for(inv.currency))
    return max(unit, unit * terms / 2) + unit / 100


def check_vat_id(vat_id: str) -> Outcome:
    """Format and check digits for the id schemes we know; NA for anything else."""
    v = vat_id.upper()
    if v.startswith("FR"):
        m = re.fullmatch(r"FR(\d{2})(\d{9})", v)
        if not m:
            return Outcome.FAIL if re.fullmatch(r"FR[0-9A-Z]{2}\d{9}", v) is None else Outcome.NA
        return Outcome.PASS if int(m.group(1)) == (12 + 3 * (int(m.group(2)) % 97)) % 97 else Outcome.FAIL
    if v.startswith("GB"):
        m = re.fullmatch(r"GB(\d{7})(\d{2})(\d{3})?", v)
        if not m:
            return Outcome.FAIL
        total = sum(int(d) * w for d, w in zip(m.group(1), range(8, 1, -1), strict=True)) + int(m.group(2))
        return Outcome.PASS if total % 97 == 0 or (total + 55) % 97 == 0 else Outcome.FAIL
    if v.startswith("IE"):
        m = re.fullmatch(r"IE(\d{7})([A-W])([A-I])?", v)
        if not m:
            return Outcome.FAIL
        total = sum(int(d) * w for d, w in zip(m.group(1), range(8, 1, -1), strict=True))
        if m.group(3):
            total += 9 * (ord(m.group(3)) - ord("A") + 1) if m.group(3) != "W" else 0
        return Outcome.PASS if "WABCDEFGHIJKLMNOPQRSTUV"[total % 23] == m.group(2) else Outcome.FAIL
    if re.fullmatch(r"\d{7}[A-Z]/[A-Z]/[A-Z]/\d{3}", v):  # Tunisian matricule fiscal: format only
        return Outcome.PASS
    if re.fullmatch(r"\d{15}", v):  # Moroccan ICE: format only
        return Outcome.PASS
    return Outcome.NA


def run_checks(inv: Invoice) -> list[Check]:
    checks: list[Check] = []
    items = inv.line_items

    for i, li in enumerate(items):
        ok = abs(li.quantity * li.unit_price - li.amount) <= _tolerance(inv)
        checks.append(
            Check(
                "line_amount",
                Outcome.PASS if ok else Outcome.FAIL,
                ("line_items",),
                "" if ok else f"line {i + 1}: {li.quantity} x {li.unit_price} != {li.amount}",
            )
        )
    if items and inv.subtotal is not None:
        total_lines = sum((li.amount for li in items), Decimal(0))
        ok = abs(total_lines - inv.subtotal) <= _tolerance(inv)
        checks.append(
            Check(
                "subtotal_vs_lines",
                Outcome.PASS if ok else Outcome.FAIL,
                ("subtotal", "line_items"),
                "" if ok else f"lines sum to {total_lines}, subtotal is {inv.subtotal}",
            )
        )
    if inv.subtotal is not None and inv.vat_rate is not None and inv.vat_amount is not None:
        expected = inv.subtotal * inv.vat_rate / 100
        ok = abs(expected - inv.vat_amount) <= _tolerance(inv)
        checks.append(
            Check(
                "vat_vs_rate",
                Outcome.PASS if ok else Outcome.FAIL,
                ("subtotal", "vat_rate", "vat_amount"),
                "" if ok else f"{inv.vat_rate}% of {inv.subtotal} is {expected:.3f}, VAT is {inv.vat_amount}",
            )
        )
    if inv.subtotal is not None and inv.vat_amount is not None and inv.total is not None:
        ok = abs(inv.subtotal + inv.vat_amount - inv.total) <= _tolerance(inv, 2)
        checks.append(
            Check(
                "total_vs_parts",
                Outcome.PASS if ok else Outcome.FAIL,
                ("subtotal", "vat_amount", "total"),
                "" if ok else f"{inv.subtotal} + {inv.vat_amount} != {inv.total}",
            )
        )
    if inv.issue_date and inv.due_date:
        ok = inv.due_date >= inv.issue_date
        checks.append(
            Check(
                "due_after_issue",
                Outcome.PASS if ok else Outcome.FAIL,
                ("issue_date", "due_date"),
                "" if ok else "due date is before the issue date",
            )
        )
    if inv.seller_vat_id:
        outcome = check_vat_id(inv.seller_vat_id)
        checks.append(
            Check(
                "vat_id_checksum",
                outcome,
                ("seller_vat_id",),
                "invalid format or check digits" if outcome is Outcome.FAIL else "",
            )
        )
    if inv.currency:
        ok = inv.currency in KNOWN_CURRENCIES
        checks.append(
            Check("currency_code", Outcome.PASS if ok else Outcome.FAIL, ("currency",), "" if ok else "unknown currency")
        )
    if inv.vat_rate is not None:
        ok = Decimal(0) <= inv.vat_rate <= Decimal(30)
        checks.append(
            Check("vat_rate_range", Outcome.PASS if ok else Outcome.FAIL, ("vat_rate",), "" if ok else "VAT rate out of range")
        )
    return checks


def issues_from(checks: list[Check]) -> list[ValidationIssue]:
    return [ValidationIssue(code=c.code, message=c.message, fields=list(c.fields)) for c in checks if c.outcome is Outcome.FAIL]


def field_validation_scores(checks: list[Check]) -> dict[str, float]:
    """1.0 if the field passes at least one check, 0.0 if it only fails checks, absent if untested."""
    passed: set[str] = set()
    failed: set[str] = set()
    for c in checks:
        if c.outcome is Outcome.PASS:
            passed.update(c.fields)
        elif c.outcome is Outcome.FAIL:
            failed.update(c.fields)
    scores = {f: 1.0 if f in passed else 0.0 for f in passed | failed}
    if "line_items" in failed:  # every line and the subtotal must agree
        scores["line_items"] = 0.0
    return scores
