from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docai.api import create_app
from docai.ocr import OcrPage
from docai.pipeline import Pipeline
from docai.store import Store

from .conftest import FakeOcr, make_pdf


@pytest.fixture
def client(tmp_path: Path, fr_page: OcrPage) -> TestClient:
    pipeline = Pipeline(ocr=FakeOcr(fr_page), denoise=False, deskew=False)
    return TestClient(create_app(pipeline, Store(tmp_path)))


def upload(client: TestClient, method: str = "rules") -> dict:  # type: ignore[type-arg]
    response = client.post("/api/documents", files={"file": ("inv.pdf", make_pdf(), "application/pdf")}, data={"method": method})
    assert response.status_code == 201, response.text
    return response.json()  # type: ignore[no-any-return]


def test_upload_extracts_and_stores(client: TestClient) -> None:
    doc = upload(client)
    fields = {f["name"]: f for f in doc["fields"]}
    assert fields["invoice_number"]["value"] == "FA-2026-0042"
    assert fields["total"]["value"] == "3000.00"
    assert client.get(f"/api/documents/{doc['id']}/pages/0").headers["content-type"] == "image/png"
    assert client.get("/api/documents").json()[0]["filename"] == "inv.pdf"


def test_rejects_non_pdf_and_unknown_method(client: TestClient) -> None:
    bad = client.post("/api/documents", files={"file": ("x.pdf", b"hello", "application/pdf")})
    assert bad.status_code == 415
    wrong = client.post("/api/documents", files={"file": ("x.pdf", make_pdf(), "application/pdf")}, data={"method": "magic"})
    assert wrong.status_code == 422


def test_review_correction_is_validated_and_recorded(client: TestClient) -> None:
    doc = upload(client)
    field = next(f for f in doc["fields"] if f["name"] == "due_date")

    bad = client.post(f"/api/fields/{field['id']}/resolve", json={"reviewer": "amina", "value": "not a date"})
    assert bad.status_code == 422

    ok = client.post(f"/api/fields/{field['id']}/resolve", json={"reviewer": "amina", "value": "2026-04-30"})
    assert ok.status_code == 200
    assert ok.json()["value"] == "2026-04-30"
    assert ok.json()["status"] == "reviewed"

    corrections = client.get("/api/corrections").json()
    assert corrections == [
        {
            **{k: corrections[0][k] for k in ("id", "field_id", "document_id", "created_at")},
            "name": "due_date",
            "action": "correct",
            "reviewer": "amina",
            "old_value": "2026-04-13",
            "new_value": "2026-04-30",
        }
    ]


def test_review_queue_and_confirm(client: TestClient, tmp_path: Path) -> None:
    doc = upload(client)
    store = Store(tmp_path)
    with store._tx() as conn:  # force one field into the queue
        conn.execute("UPDATE fields SET status = 'needs_review', confidence = 0.3 WHERE document_id = ? AND name = 'total'", (doc["id"],))
    queue = client.get("/api/review").json()
    assert [f["name"] for f in queue] == ["total"]
    confirmed = client.post(f"/api/fields/{queue[0]['id']}/resolve", json={"reviewer": "amina", "action": "confirm"})
    assert confirmed.json()["value"] == "3000.00"
    assert client.get("/api/review").json() == []


def test_line_items_correction_must_match_schema(client: TestClient) -> None:
    doc = upload(client)
    field = next(f for f in doc["fields"] if f["name"] == "line_items")
    bad = client.post(f"/api/fields/{field['id']}/resolve", json={"reviewer": "a", "value": [{"description": "x"}]})
    assert bad.status_code == 422
    good = [{"description": "x", "quantity": "1", "unit_price": "2.50", "amount": "2.50"}]
    assert client.post(f"/api/fields/{field['id']}/resolve", json={"reviewer": "a", "value": good}).status_code == 200


def test_static_frontend_is_served(client: TestClient) -> None:
    assert "docai" in client.get("/").text
