"""OCR with RapidOCR (PaddleOCR PP-OCRv5 models running on ONNX Runtime, CPU only).

The Arabic recogniser also reads Latin script, the Latin one is better on accents. In `auto` mode we
run the Arabic model first and switch to the Latin model when the page has (almost) no Arabic."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Literal, Protocol

from .imaging import Gray
from .parsing import has_arabic

Lang = Literal["auto", "latin", "arabic"]


@dataclass(frozen=True)
class OcrLine:
    text: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class OcrPage:
    lines: list[OcrLine]
    width: int
    height: int
    script: Literal["latin", "arabic"]
    rows: list[list[OcrLine]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.rows:
            self.rows = group_rows(self.lines, rtl=self.script == "arabic")

    def text(self) -> str:
        return "\n".join("   ".join(cell.text for cell in row) for row in self.rows)


class OcrEngine(Protocol):
    def __call__(self, image: Gray, lang: Lang = "auto") -> OcrPage: ...


def group_rows(lines: list[OcrLine], rtl: bool) -> list[list[OcrLine]]:
    """Group boxes into visual rows (vertical-centre proximity), cells in reading order."""
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda ln: ln.cy):
        if rows:
            last = rows[-1]
            ref = sum(c.cy for c in last) / len(last)
            tol = 0.55 * max(line.height, sum(c.height for c in last) / len(last))
            if abs(line.cy - ref) <= tol:
                last.append(line)
                continue
        rows.append([line])
    return [sorted(row, key=lambda c: -c.box[2] if rtl else c.box[0]) for row in rows]


def arabic_ratio(lines: list[OcrLine]) -> float:
    letters = sum(sum(c.isalpha() for c in ln.text) for ln in lines)
    arabic = sum(sum(has_arabic(c) for c in ln.text) for ln in lines)
    return arabic / letters if letters else 0.0


@cache
def _engine(script: str) -> Any:
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

    return RapidOCR(
        params={
            "Global.log_level": "error",
            "Rec.lang_type": LangRec.ARABIC if script == "arabic" else LangRec.LATIN,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
        }
    )


class RapidOcr:
    """Thread-safe wrapper; ONNX sessions are shared, calls are serialised."""

    def __init__(self, arabic_threshold: float = 0.05) -> None:
        self.arabic_threshold = arabic_threshold
        self._lock = threading.Lock()

    def _run(self, image: Gray, script: str) -> list[OcrLine]:
        with self._lock:
            result = _engine(script)(image)
        if result.boxes is None:
            return []
        lines = []
        for box, text, score in zip(result.boxes, result.txts, result.scores, strict=True):
            xs, ys = [float(p[0]) for p in box], [float(p[1]) for p in box]
            lines.append(OcrLine(text=str(text), score=float(score), box=(min(xs), min(ys), max(xs), max(ys))))
        return lines

    def __call__(self, image: Gray, lang: Lang = "auto") -> OcrPage:
        h, w = image.shape
        if lang == "latin":
            return OcrPage(self._run(image, "latin"), w, h, "latin")
        lines = self._run(image, "arabic")
        if lang == "arabic" or arabic_ratio(lines) >= self.arabic_threshold:
            return OcrPage(lines, w, h, "arabic")
        return OcrPage(self._run(image, "latin"), w, h, "latin")
