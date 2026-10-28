from __future__ import annotations

from decimal import Decimal

import pytest

from docai.schema import Invoice
from docai.validate import Outcome, check_vat_id, field_validation_scores, issues_from, run_checks


def test_consistent_invoice_passes_everything(fr_invoice: Invoice) -> None:
    checks = run_checks(fr_invoice)
    assert {c.code for c in checks} >= {"line_amount", "subtotal_vs_lines", "vat_vs_rate", "total_vs_parts", "vat_id_checksum"}
    assert issues_from(checks) == []
    assert all(v == 1.0 for v in field_validation_scores(checks).values())


def test_wrong_total_is_blamed_on_total_only(fr_invoice: Invoice) -> None:
    fr_invoice.total = Decimal("3100.00")
    checks = run_checks(fr_invoice)
    assert [i.code for i in issues_from(checks)] == ["total_vs_parts"]
    scores = field_validation_scores(checks)
    assert scores["total"] == 0.0
    assert scores["subtotal"] == 1.0  # still corroborated by the line items and the VAT rate
    assert scores["vat_amount"] == 1.0


def test_bad_line_fails_line_items(fr_invoice: Invoice) -> None:
    fr_invoice.line_items[0].amount = Decimal("2100.00")
    scores = field_validation_scores(run_checks(fr_invoice))
    assert scores["line_items"] == 0.0


def test_three_decimal_currency_tolerance() -> None:
    inv = Invoice(
        currency="TND", subtotal=Decimal("288.760"), vat_rate=Decimal(19), vat_amount=Decimal("54.864"), total=Decimal("343.624")
    )
    assert issues_from(run_checks(inv)) == []


def test_due_before_issue_is_flagged(fr_invoice: Invoice) -> None:
    fr_invoice.due_date, fr_invoice.issue_date = fr_invoice.issue_date, fr_invoice.due_date
    assert "due_after_issue" in {i.code for i in issues_from(run_checks(fr_invoice))}


@pytest.mark.parametrize(
    ("vat_id", "expected"),
    [
        ("FR40303265045", Outcome.PASS),
        ("FR41303265045", Outcome.FAIL),
        ("GB863724896", Outcome.PASS),  # from the generator, valid mod-97
        ("GB863724897", Outcome.FAIL),
        ("IE6388047V", Outcome.PASS),
        ("IE6388047W", Outcome.FAIL),
        ("1234567A/M/A/000", Outcome.PASS),
        ("001234567890123", Outcome.PASS),
        ("US123", Outcome.NA),
    ],
)
def test_vat_id_checks(vat_id: str, expected: Outcome) -> None:
    assert check_vat_id(vat_id) is expected
