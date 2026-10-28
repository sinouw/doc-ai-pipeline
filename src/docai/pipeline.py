"""End-to-end: PDF -> pages -> clean-up -> OCR -> extraction -> validation -> per-field confidence."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .confidence import DEFAULT_THRESHOLD, score_fields
from .extract import extract_llm
from .imaging import Gray, preprocess, render_pdf
from .llm import GroqClient
from .ocr import Lang, OcrEngine, OcrPage, RapidOcr
from .rules import extract_rules
from .schema import DocumentResult, Invoice, Usage
from .validate import issues_from, run_checks

Method = Literal["rules", "ocr_llm", "vlm", "hybrid"]
METHODS: tuple[Method, ...] = ("rules", "ocr_llm", "vlm", "hybrid")


@dataclass
class PreparedDocument:
    """Everything that does not depend on the extraction method, so evaluation can reuse it."""

    source: str
    pages: list[Gray]
    ocr: list[OcrPage]
    rules: Invoice
    skew_degrees: list[float] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def ocr_text(self) -> str:
        return "\n\n".join(p.text() for p in self.ocr)

    @property
    def script(self) -> str:
        return self.ocr[0].script if self.ocr else "latin"


def _merge_pages(pages: list[Invoice]) -> Invoice:
    """First non-empty value per field across pages; line items concatenated."""
    merged = Invoice()
    for inv in pages:
        for name in Invoice.model_fields:
            if name == "line_items":
                merged.line_items.extend(inv.line_items)
            elif getattr(merged, name) is None:
                setattr(merged, name, getattr(inv, name))
    return merged


class Pipeline:
    def __init__(
        self,
        ocr: OcrEngine | None = None,
        client: GroqClient | None = None,
        *,
        dpi: int = 200,
        max_pages: int = 5,
        threshold: float = DEFAULT_THRESHOLD,
        denoise: bool = True,
        deskew: bool = True,
    ) -> None:
        self.ocr = ocr or RapidOcr()
        self._client = client
        self.dpi = dpi
        self.max_pages = max_pages
        self.threshold = threshold
        self.denoise = denoise
        self.deskew = deskew

    @property
    def client(self) -> GroqClient:
        if self._client is None:
            self._client = GroqClient()
        return self._client

    def prepare(self, source: Path | bytes, name: str = "upload.pdf", lang: Lang = "auto") -> PreparedDocument:
        t0 = time.perf_counter()
        raw_pages = render_pdf(source, dpi=self.dpi, max_pages=self.max_pages)
        t1 = time.perf_counter()
        cleaned = [preprocess(p, do_denoise=self.denoise, do_deskew=self.deskew) for p in raw_pages]
        pages = [c.image for c in cleaned]
        t2 = time.perf_counter()
        ocr_pages = [self.ocr(p, lang) for p in pages]
        t3 = time.perf_counter()
        rules = _merge_pages([extract_rules(p) for p in ocr_pages])
        t4 = time.perf_counter()
        return PreparedDocument(
            source=name,
            pages=pages,
            ocr=ocr_pages,
            rules=rules,
            skew_degrees=[c.skew_degrees for c in cleaned],
            timings_ms={
                "render": (t1 - t0) * 1000,
                "preprocess": (t2 - t1) * 1000,
                "ocr": (t3 - t2) * 1000,
                "rules": (t4 - t3) * 1000,
            },
        )

    def extract(self, doc: PreparedDocument, method: Method) -> DocumentResult:
        timings = dict(doc.timings_ms)
        usage = Usage()
        t0 = time.perf_counter()
        if method == "rules":
            invoice, reference = doc.rules, None
        else:
            invoice, usage = extract_llm(self.client, method, doc.pages, doc.ocr_text)
            reference = doc.rules
        timings["extract"] = (time.perf_counter() - t0) * 1000
        checks = run_checks(invoice)
        fields = score_fields(invoice, checks, doc.ocr, reference, self.threshold)
        timings["total"] = sum(timings.values())
        return DocumentResult(
            source=doc.source,
            method=method,
            pages=len(doc.pages),
            language=doc.script,
            invoice=invoice,
            fields=fields,
            issues=issues_from(checks),
            usage=usage,
            timings_ms={k: round(v, 1) for k, v in timings.items()},
            ocr_text=doc.ocr_text,
        )

    def process(
        self, source: Path | bytes, method: Method = "hybrid", name: str = "upload.pdf", lang: Lang = "auto"
    ) -> tuple[DocumentResult, PreparedDocument]:
        doc = self.prepare(source, name=name, lang=lang)
        return self.extract(doc, method), doc
