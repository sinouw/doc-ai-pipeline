from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from docai.evaluation import score_record, summarize
from docai.imaging import estimate_skew, preprocess, render_pdf, rotate
from docai.schema import Invoice

from .conftest import make_pdf


def test_score_record_field_by_field(fr_invoice: Invoice) -> None:
    label = fr_invoice.model_dump(mode="json")
    prediction = {**label, "buyer_name": "dupont  &  FILS", "total": "3000", "invoice_number": "FA-2026-0043"}
    correct = score_record(prediction, label)
    assert correct["buyer_name"] is True  # case and spacing do not matter
    assert correct["total"] is True  # 3000 == 3000.00
    assert correct["invoice_number"] is False


def test_summarize_routing_and_cost(fr_invoice: Invoice) -> None:
    label = fr_invoice.model_dump(mode="json")
    fields = [{"name": n, "confidence": 0.9, "signals": {}} for n in score_record(label, label)]
    fields[0] = {"name": "invoice_number", "confidence": 0.4, "signals": {}}
    record = {
        "method": "vlm",
        "lang": "fr",
        "severity": "light",
        "injected_error": None,
        "error": None,
        "prediction": {**label, "invoice_number": "WRONG"},
        "label": label,
        "correct": score_record({**label, "invoice_number": "WRONG"}, label),
        "fields": fields,
        "issues": [],
        "usage": {"prompt_tokens": 2000, "completion_tokens": 500, "calls": 1, "retries": 0, "wait_ms": 1000.0},
        "timings_ms": {"total": 5000.0, "extract": 3000.0},
        "pages": 1,
    }
    out = summarize([record], {"vlm": (0.8, 4.0)}, (0.8,))["vlm"]
    assert out["field_accuracy"] == round(100 * 11 / 12, 1)
    assert out["exact_match"] == 0.0
    assert out["cost_usd_per_page"] == 0.0036  # 2000 * 0.8e-6 + 500 * 4e-6
    assert out["latency_excl_rate_limit_ms_per_page"]["p50"] == 4000
    assert out["review_routing"]["0.8"] == {
        "review_rate": 8.3,
        "auto_accepted_accuracy": 100.0,
        "errors_caught": 100.0,
        "errors_missed": 0,
    }


def _text_page() -> np.ndarray:
    img = Image.new("L", (1000, 1000), 255)
    draw = ImageDraw.Draw(img)
    for y in range(100, 900, 40):
        draw.rectangle((100, y, 900, y + 14), fill=0)  # text-line-like bars
    return np.asarray(img, dtype=np.uint8)


def test_deskew_recovers_rotation() -> None:
    page = _text_page()
    skewed = rotate(page, 2.5)
    assert abs(estimate_skew(skewed) + 2.5) <= 0.2
    result = preprocess(skewed, do_denoise=False)
    assert abs(result.skew_degrees + 2.5) <= 0.2


def test_render_pdf_returns_grayscale_pages() -> None:
    pages = render_pdf(make_pdf(), dpi=100)
    assert len(pages) == 1
    assert pages[0].ndim == 2 and pages[0].dtype == np.uint8
