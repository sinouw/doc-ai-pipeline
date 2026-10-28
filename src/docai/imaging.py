"""PDF rasterisation and scan clean-up (denoise, deskew)."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pypdfium2 as pdfium
from PIL import Image

Gray = npt.NDArray[np.uint8]


def render_pdf(source: Path | bytes, dpi: int = 200, max_pages: int = 10) -> list[Gray]:
    """Rasterise up to `max_pages` pages to 8-bit grayscale arrays."""
    pdf = pdfium.PdfDocument(source)
    try:
        pages = []
        for index in range(min(len(pdf), max_pages)):
            bitmap = pdf[index].render(scale=dpi / 72, grayscale=True)
            pages.append(np.asarray(bitmap.to_pil().convert("L"), dtype=np.uint8).copy())
        return pages
    finally:
        pdf.close()


def count_pages(source: Path | bytes) -> int:
    pdf = pdfium.PdfDocument(source)
    try:
        return len(pdf)
    finally:
        pdf.close()


@dataclass(frozen=True)
class Preprocessed:
    image: Gray
    skew_degrees: float


def denoise(image: Gray) -> Gray:
    """Median filter removes salt-and-pepper speckle; NL-means smooths sensor-like noise."""
    out = cv2.medianBlur(image, 3)
    return np.asarray(cv2.fastNlMeansDenoising(out, None, h=12, templateWindowSize=7, searchWindowSize=15), dtype=np.uint8)


def estimate_skew(image: Gray, max_angle: float = 6.0) -> float:
    """Angle (degrees) that makes text lines horizontal, found by maximising the variance of the
    horizontal projection profile. Coarse 0.5° search, then a 0.1° refinement, on a downscaled copy."""
    scale = 1000 / max(image.shape)
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    center = (w / 2, h / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rotated.sum(axis=1, dtype=np.float64)
        return float(np.var(profile))

    coarse = max(np.arange(-max_angle, max_angle + 0.01, 0.5), key=score)
    fine = max(np.arange(coarse - 0.5, coarse + 0.51, 0.1), key=score)
    return round(float(fine), 2)


def rotate(image: Gray, angle: float) -> Gray:
    h, w = image.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return np.asarray(
        cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255),
        dtype=np.uint8,
    )


def preprocess(image: Gray, *, do_denoise: bool = True, do_deskew: bool = True) -> Preprocessed:
    out = denoise(image) if do_denoise else image
    angle = estimate_skew(out) if do_deskew else 0.0
    if abs(angle) >= 0.1:
        out = rotate(out, angle)
    return Preprocessed(image=out, skew_degrees=angle)


def to_png_bytes(image: Gray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def to_jpeg_data_url(image: Gray, max_side: int = 1600, quality: int = 85) -> str:
    pil = Image.fromarray(image)
    pil.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
