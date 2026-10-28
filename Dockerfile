FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    DOCAI_DATA_DIR=/data

# OpenCV and ONNX Runtime need these at runtime on slim images.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

# Fetch every OCR model at build time (recognisers load lazily, so run one real pass per script).
RUN uv run --no-sync python -c "import numpy as np, cv2; from docai.ocr import RapidOcr; \
img = np.full((120, 600), 255, np.uint8); cv2.putText(img, 'INVOICE 2026', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3); \
ocr = RapidOcr(); print(ocr(img, 'arabic').text(), '|', ocr(img, 'latin').text())"

RUN useradd --create-home app && mkdir -p /data && chown app /data
USER app
EXPOSE 8000
VOLUME ["/data"]
CMD ["uv", "run", "--no-sync", "docai", "serve", "--host", "0.0.0.0", "--port", "8000"]
