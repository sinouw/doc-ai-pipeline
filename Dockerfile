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

# Fetch the OCR models at build time so the container starts offline-ready.
RUN uv run python -c "from docai.ocr import _engine; _engine('arabic'); _engine('latin')"

RUN useradd --create-home app && mkdir -p /data && chown app /data
USER app
EXPOSE 8000
VOLUME ["/data"]
CMD ["uv", "run", "--no-sync", "docai", "serve", "--host", "0.0.0.0", "--port", "8000"]
