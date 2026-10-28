"""Typed invoice schema shared by the extractors, the validator, the store and the evaluation."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    quantity: Decimal
    unit_price: Decimal
    amount: Decimal


class Invoice(BaseModel):
    """What we extract from one invoice. Every field is optional: a missing value is a result, not a crash."""

    model_config = ConfigDict(extra="forbid")

    invoice_number: str | None = None
    issue_date: date | None = None
    due_date: date | None = None
    seller_name: str | None = None
    seller_vat_id: str | None = None
    buyer_name: str | None = None
    currency: str | None = Field(default=None, description="ISO 4217 code, e.g. EUR, TND, MAD")
    subtotal: Decimal | None = Field(default=None, description="Total before tax (HT)")
    vat_rate: Decimal | None = Field(default=None, description="Percent, e.g. 20 for 20%")
    vat_amount: Decimal | None = None
    total: Decimal | None = Field(default=None, description="Total including tax (TTC)")
    line_items: list[LineItem] = Field(default_factory=list)


# Fields reviewed and scored one by one. line_items is reviewed as a single unit.
SCALAR_FIELDS: tuple[str, ...] = (
    "invoice_number",
    "issue_date",
    "due_date",
    "seller_name",
    "seller_vat_id",
    "buyer_name",
    "currency",
    "subtotal",
    "vat_rate",
    "vat_amount",
    "total",
)
REVIEW_FIELDS: tuple[str, ...] = (*SCALAR_FIELDS, "line_items")


class FieldStatus(StrEnum):
    AUTO_ACCEPTED = "auto_accepted"
    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"


class FieldResult(BaseModel):
    name: str
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    signals: dict[str, float] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    status: FieldStatus = FieldStatus.NEEDS_REVIEW


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    retries: int = 0
    wait_ms: float = 0.0  # time spent backing off on rate limits, excluded from service latency

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls
        self.retries += other.retries
        self.wait_ms += other.wait_ms


class ValidationIssue(BaseModel):
    code: str
    message: str
    fields: list[str]


class DocumentResult(BaseModel):
    source: str
    method: str
    pages: int
    language: str | None
    invoice: Invoice
    fields: list[FieldResult]
    issues: list[ValidationIssue]
    usage: Usage
    timings_ms: dict[str, float]
    ocr_text: str = ""
