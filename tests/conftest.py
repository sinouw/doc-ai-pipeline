from __future__ import annotations

import io
from decimal import Decimal

import numpy as np
import pytest
from PIL import Image, ImageDraw

from docai.imaging import Gray
from docai.ocr import Lang, OcrLine, OcrPage
from docai.schema import Invoice, LineItem


def line(text: str, x0: float, y0: float, x1: float | None = None, score: float = 0.99) -> OcrLine:
    return OcrLine(text=text, score=score, box=(x0, y0, x1 if x1 is not None else x0 + 12 * len(text), y0 + 30))


FR_LINES = [
    line("ACME Conseil SARL", 100, 80),
    line("FACTURE", 900, 80),
    line("N° TVA: FR40303265045", 100, 150),
    line("Facture N° :", 100, 260),
    line("FA-2026-0042", 360, 260),
    line("Client", 1000, 260),
    line("Date :", 100, 300),
    line("14/03/2026", 360, 300),
    line("Dupont & Fils", 980, 300),
    line("Échéance :", 100, 340),
    line("13/04/2026", 360, 340),
    line("Désignation", 100, 470),
    line("Qté", 700, 470),
    line("Prix unitaire", 820, 470),
    line("Montant", 1050, 470),
    line("Audit sécurité", 100, 520),
    line("2", 720, 520),
    line("1 000,00", 860, 520),
    line("2 000,00", 1060, 520),
    line("Formation", 100, 564),
    line("1", 720, 564),
    line("500,00", 880, 564),
    line("500,00", 1080, 564),
    line("Total HT", 640, 660),
    line("2 500,00 EUR", 1000, 660),
    line("TVA 20%", 640, 704),
    line("500,00 EUR", 1020, 704),
    line("Total TTC", 640, 748),
    line("3 000,00 EUR", 1000, 748),
]


@pytest.fixture
def fr_page() -> OcrPage:
    return OcrPage(lines=list(FR_LINES), width=1240, height=1754, script="latin")


@pytest.fixture
def fr_invoice() -> Invoice:
    return Invoice(
        invoice_number="FA-2026-0042",
        issue_date="2026-03-14",  # type: ignore[arg-type]
        due_date="2026-04-13",  # type: ignore[arg-type]
        seller_name="ACME Conseil SARL",
        seller_vat_id="FR40303265045",
        buyer_name="Dupont & Fils",
        currency="EUR",
        subtotal=Decimal("2500.00"),
        vat_rate=Decimal("20"),
        vat_amount=Decimal("500.00"),
        total=Decimal("3000.00"),
        line_items=[
            LineItem(description="Audit sécurité", quantity=Decimal(2), unit_price=Decimal("1000.00"), amount=Decimal("2000.00")),
            LineItem(description="Formation", quantity=Decimal(1), unit_price=Decimal("500.00"), amount=Decimal("500.00")),
        ],
    )


class FakeOcr:
    def __init__(self, page: OcrPage) -> None:
        self.page = page

    def __call__(self, image: Gray, lang: Lang = "auto") -> OcrPage:
        return self.page


def make_pdf(text: str = "FACTURE") -> bytes:
    img = Image.new("L", (620, 877), 255)
    ImageDraw.Draw(img).text((50, 50), text, fill=0)
    buf = io.BytesIO()
    img.save(buf, "PDF", resolution=75)
    return buf.getvalue()


def blank_page() -> Gray:
    return np.full((200, 200), 255, dtype=np.uint8)
