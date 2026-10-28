"""Runs the real RapidOCR models on one committed dev scan (models download on first use)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docai.compare import values_equal
from docai.pipeline import Pipeline

DEV = Path(__file__).resolve().parents[1] / "data" / "dev"


@pytest.mark.ocr
def test_rules_on_a_real_scan() -> None:
    label = json.loads((DEV / "fr-001.json").read_text(encoding="utf-8"))["label"]
    result, doc = Pipeline().process(DEV / "fr-001.pdf", method="rules")
    assert doc.script == "latin"
    for name in ("invoice_number", "issue_date", "total", "currency"):
        assert values_equal(name, getattr(result.invoice, name), label[name]), name
