from __future__ import annotations

from decimal import Decimal

from docai.confidence import score_fields
from docai.ocr import OcrPage, group_rows
from docai.rules import extract_rules
from docai.schema import FieldStatus, Invoice
from docai.validate import run_checks

from .conftest import line


def test_rules_extract_french_invoice(fr_page: OcrPage, fr_invoice: Invoice) -> None:
    assert extract_rules(fr_page) == fr_invoice


def test_rules_read_arabic_rows_right_to_left() -> None:
    page = OcrPage(
        lines=[
            line("شركة النور للخدمات", 900, 80),
            line("فاتورة", 100, 80),
            line("رقم الفاتورة", 1000, 260),
            line("F-2026-17", 700, 260),
            line("التاريخ", 1050, 300),
            line("2026/05/30", 700, 300),
            line("المجموع دون الأداءات", 900, 660),
            line("288,760 TND", 200, 660),
            line("الأداء على القيمة المضافة 19%", 850, 704),
            line("54,864 TND", 200, 704),
            line("المجموع الجملي", 950, 748),
            line("343,624 TND", 200, 748),
        ],
        width=1240,
        height=1754,
        script="arabic",
    )
    inv = extract_rules(page)
    assert inv.seller_name == "شركة النور للخدمات"
    assert inv.invoice_number == "F-2026-17"
    assert str(inv.issue_date) == "2026-05-30"
    assert inv.currency == "TND"
    assert (inv.subtotal, inv.vat_rate, inv.vat_amount, inv.total) == (
        Decimal("288.760"),
        Decimal(19),
        Decimal("54.864"),
        Decimal("343.624"),
    )


def test_group_rows_orders_cells_by_script() -> None:
    a, b = line("left", 100, 100), line("right", 800, 104)
    assert [c.text for c in group_rows([b, a], rtl=False)[0]] == ["left", "right"]
    assert [c.text for c in group_rows([a, b], rtl=True)[0]] == ["right", "left"]


def test_confident_fields_are_auto_accepted(fr_page: OcrPage, fr_invoice: Invoice) -> None:
    fields = score_fields(fr_invoice, run_checks(fr_invoice), [fr_page], reference=fr_invoice)
    assert all(f.status is FieldStatus.AUTO_ACCEPTED for f in fields), [(f.name, f.confidence) for f in fields]


def test_disagreeing_unsupported_value_goes_to_review(fr_page: OcrPage, fr_invoice: Invoice) -> None:
    llm = fr_invoice.model_copy(update={"invoice_number": "FA-2026-0024", "total": Decimal("3100.00")})
    fields = {f.name: f for f in score_fields(llm, run_checks(llm), [fr_page], reference=fr_invoice)}
    assert fields["invoice_number"].status is FieldStatus.NEEDS_REVIEW
    assert fields["invoice_number"].signals == {"ocr_support": 0.0, "agreement": 0.0}
    assert fields["total"].status is FieldStatus.NEEDS_REVIEW
    assert fields["total"].issues
    assert fields["subtotal"].status is FieldStatus.AUTO_ACCEPTED


def test_missing_field_has_zero_confidence(fr_page: OcrPage) -> None:
    fields = {f.name: f for f in score_fields(Invoice(), [], [fr_page], reference=None)}
    assert fields["total"].confidence == 0.0
    assert "missing" in fields["total"].issues
