from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from docai.parsing import find_dates, normalize_currency, normalize_id, normalize_text, parse_amount, parse_date


@pytest.mark.parametrize(
    ("text", "decimals", "expected"),
    [
        ("1 234,56", 2, "1234.56"),
        ("1,234.56", 2, "1234.56"),
        ("1.234,56", 2, "1234.56"),
        ("5 912,90 EUR", 2, "5912.90"),
        ("1234.5", 2, "1234.5"),
        ("1,234", 2, "1234"),  # thousands separator for 2-decimal currencies
        ("288,760", 3, "288.760"),  # Tunisian dinar: three decimals
        ("18 400,000 TND", 3, "18400.000"),
        ("٤٥٠٫٠٠", 2, "450.00"),  # Arabic-Indic digits and decimal separator
        ("12,5", 2, "12.5"),
    ],
)
def test_parse_amount(text: str, decimals: int, expected: str) -> None:
    assert parse_amount(text, decimals) == Decimal(expected)


def test_parse_amount_rejects_text() -> None:
    assert parse_amount("Total TTC") is None
    assert parse_amount(None) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("14/03/2026", date(2026, 3, 14)),
        ("2026/05/30", date(2026, 5, 30)),
        ("2026-03-14", date(2026, 3, 14)),
        ("29 janvier 2026", date(2026, 1, 29)),
        ("1er août 2026", date(2026, 8, 1)),
        ("March 14, 2026", date(2026, 3, 14)),
        ("21 Jun 2026", date(2026, 6, 21)),
        ("١٤/٠٣/٢٠٢٦", date(2026, 3, 14)),
        ("Date : 31/02/2026", None),
    ],
)
def test_parse_date(text: str, expected: date | None) -> None:
    assert parse_date(text) == expected


def test_find_dates_keeps_order_and_dedupes() -> None:
    assert find_dates("du 01/02/2026 au 15/02/2026, émis le 01/02/2026") == [date(2026, 2, 1), date(2026, 2, 15)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("€", "EUR"), ("1 200,00 EUR", "EUR"), ("£", "GBP"), ("DT", "TND"), ("د.ت", "TND"), ("dh", "MAD"), ("XYZ", None)],
)
def test_normalize_currency(text: str, expected: str | None) -> None:
    assert normalize_currency(text) == expected


def test_normalize_id_and_text() -> None:
    assert normalize_id("fr 40 303.265.045") == "FR40303265045"
    assert normalize_text("  ACME   Conseil, S.A.R.L. ") == "acme conseil s a r l"
    assert normalize_text("شَرِكة") == normalize_text("شركة")
