"""FastAPI app: upload PDFs, see extracted fields, review low-confidence ones."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from .imaging import to_png_bytes
from .pipeline import METHODS, Pipeline
from .schema import Invoice
from .store import Store

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class Resolution(BaseModel):
    reviewer: str = Field(min_length=1, max_length=80)
    value: Any = None
    action: str = Field(default="correct", pattern="^(correct|confirm)$")


def coerce_field(name: str, value: Any) -> Any:
    """Validate a reviewer's value against the schema type of that field and return it JSON-ready."""
    if name not in Invoice.model_fields:
        raise ValueError(f"unknown field {name}")
    if value in ("", None):
        return None
    invoice = Invoice.model_validate({name: value})
    return invoice.model_dump(mode="json")[name]


def create_app(pipeline: Pipeline | None = None, store: Store | None = None) -> FastAPI:
    app = FastAPI(title="docai", version="0.1.0")
    state_pipeline = pipeline or Pipeline()
    state_store = store or Store(Path(os.environ.get("DOCAI_DATA_DIR", "var")))

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/documents", status_code=201)
    async def upload(
        file: Annotated[UploadFile, File()],
        method: Annotated[str, Form()] = "hybrid",
    ) -> dict[str, Any]:
        if method not in METHODS:
            raise HTTPException(422, f"method must be one of {', '.join(METHODS)}")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file too large (20 MB max)")
        if not data.startswith(b"%PDF"):
            raise HTTPException(415, "only PDF files are accepted")
        try:
            result, doc = await run_in_threadpool(state_pipeline.process, data, method, file.filename or "upload.pdf")
        except Exception as exc:  # surface pipeline failures (bad PDF, LLM outage) as 502 with a reason
            raise HTTPException(502, f"processing failed: {exc}") from exc
        doc_id = state_store.save(result, [to_png_bytes(p) for p in doc.pages])
        document = state_store.get_document(doc_id)
        assert document is not None
        return document

    @app.get("/api/documents")
    def documents() -> list[dict[str, Any]]:
        return state_store.list_documents()

    @app.get("/api/documents/{doc_id}")
    def document(doc_id: int) -> dict[str, Any]:
        doc = state_store.get_document(doc_id)
        if doc is None:
            raise HTTPException(404, "document not found")
        return doc

    @app.get("/api/documents/{doc_id}/pages/{page}")
    def page_image(doc_id: int, page: int) -> FileResponse:
        path = state_store.page_path(doc_id, page)
        if path is None:
            raise HTTPException(404, "page not found")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/review")
    def review_queue(limit: int = 100) -> list[dict[str, Any]]:
        return state_store.review_queue(limit)

    @app.post("/api/fields/{field_id}/resolve")
    def resolve(field_id: int, body: Resolution) -> dict[str, Any]:
        field = state_store.get_field(field_id)
        if field is None:
            raise HTTPException(404, "field not found")
        value = None
        if body.action == "correct":
            try:
                value = coerce_field(field["name"], body.value)
            except (ValidationError, ValueError) as exc:
                raise HTTPException(422, f"invalid value for {field['name']}: {exc}") from exc
        return state_store.resolve(field_id, value, body.reviewer, body.action)

    @app.get("/api/corrections")
    def corrections() -> list[dict[str, Any]]:
        return state_store.corrections()

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
