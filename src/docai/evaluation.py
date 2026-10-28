"""Metrics over evaluation records (one record = one document processed by one method)."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from .compare import line_items_equal, values_equal
from .confidence import needs_review
from .schema import REVIEW_FIELDS, Invoice

ARITHMETIC_ISSUES = {"subtotal_vs_lines", "vat_vs_rate", "total_vs_parts", "line_amount"}


def score_record(prediction: dict[str, Any], label: dict[str, Any]) -> dict[str, bool]:
    pred, gold = Invoice.model_validate(prediction), Invoice.model_validate(label)
    return {name: values_equal(name, getattr(pred, name), getattr(gold, name)) for name in REVIEW_FIELDS}


def _pct(numerator: float, denominator: float) -> float | None:
    return round(100 * numerator / denominator, 1) if denominator else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q * 100) - 1]


def summarize(
    records: list[dict[str, Any]], prices: dict[str, tuple[float, float]], thresholds: tuple[float, ...]
) -> dict[str, Any]:
    """`records` carry: method, lang, severity, injected_error, correct (field->bool), fields (list of
    FieldResult dicts), issues, usage, timings_ms, pages, prediction, label, error."""
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_method[r["method"]].append(r)
    out: dict[str, Any] = {}
    for method, rows in by_method.items():
        ok_rows = [r for r in rows if not r.get("error")]
        field_acc = {name: _pct(sum(r["correct"][name] for r in ok_rows), len(rows)) for name in REVIEW_FIELDS}
        total_fields = len(rows) * len(REVIEW_FIELDS)
        correct_fields = sum(sum(r["correct"].values()) for r in ok_rows)
        exact = sum(all(r["correct"].values()) for r in ok_rows)
        groups: dict[str, dict[str, Any]] = {}
        for key in ("lang", "severity"):
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in rows:
                buckets[r[key]].append(r)
            groups[key] = {
                k: {
                    "docs": len(v),
                    "field_accuracy": _pct(
                        sum(sum(r["correct"].values()) for r in v if not r.get("error")), len(v) * len(REVIEW_FIELDS)
                    ),
                    "exact_match": _pct(sum(all(r["correct"].values()) for r in v if not r.get("error")), len(v)),
                }
                for k, v in sorted(buckets.items())
            }
        descriptions = [
            line_items_equal(r["prediction"]["line_items"], r["label"]["line_items"], with_description=True) for r in ok_rows
        ]
        pages = sum(r["pages"] for r in ok_rows) or 1
        totals = [r["timings_ms"]["total"] / r["pages"] for r in ok_rows]
        extract = [r["timings_ms"]["extract"] / r["pages"] for r in ok_rows]
        wait = [r["usage"].get("wait_ms", 0.0) / r["pages"] for r in ok_rows]
        service = [t - w for t, w in zip(totals, wait, strict=True)]
        prompt = sum(r["usage"]["prompt_tokens"] for r in ok_rows)
        completion = sum(r["usage"]["completion_tokens"] for r in ok_rows)
        price_in, price_out = prices.get(method, (0.0, 0.0))
        cost = (prompt * price_in + completion * price_out) / 1_000_000

        # Review routing: what the confidence threshold sends to a human, and what it lets through.
        routing = {}
        for t in thresholds:
            reviewed = auto = auto_ok = errors = errors_caught = 0
            for r in ok_rows:
                flagged_fields = {f["name"] for f in r["fields"] if needs_review(f["confidence"], f["signals"], t)}
                for name in REVIEW_FIELDS:
                    correct = r["correct"][name]
                    errors += not correct
                    if name in flagged_fields:
                        reviewed += 1
                        errors_caught += not correct
                    else:
                        auto += 1
                        auto_ok += correct
            n = reviewed + auto
            routing[str(t)] = {
                "review_rate": _pct(reviewed, n),
                "auto_accepted_accuracy": _pct(auto_ok, auto),
                "errors_caught": _pct(errors_caught, errors),
                "errors_missed": errors - errors_caught,
            }

        injected = [r for r in ok_rows if r["injected_error"]]
        clean = [r for r in ok_rows if not r["injected_error"]]
        out[method] = {
            "docs": len(rows),
            "errors": len(rows) - len(ok_rows),
            "field_accuracy": _pct(correct_fields, total_fields),
            "exact_match": _pct(exact, len(rows)),
            "field_accuracy_by_field": field_acc,
            "line_items_with_descriptions": _pct(sum(descriptions), len(rows)),
            "by": groups,
            "latency_ms_per_page": {
                "p50": _round(_quantile(totals, 0.5)),
                "p95": _round(_quantile(totals, 0.95)),
                "mean": _round(statistics.fmean(totals) if totals else None),
            },
            # Wall clock minus time spent sleeping on HTTP 429 back-offs (the free tier limit is shared).
            "latency_excl_rate_limit_ms_per_page": {
                "p50": _round(_quantile(service, 0.5)),
                "p95": _round(_quantile(service, 0.95)),
            },
            "extract_ms_per_page": {"p50": _round(_quantile(extract, 0.5)), "p95": _round(_quantile(extract, 0.95))},
            "rate_limit_wait_ms_total": _round(sum(wait)),
            "tokens_per_page": {"prompt": round(prompt / pages), "completion": round(completion / pages)},
            "llm_calls": sum(r["usage"]["calls"] for r in ok_rows),
            "rate_limit_retries": sum(r["usage"]["retries"] for r in ok_rows),
            "cost_usd_per_page": round(cost / pages, 6),
            "cost_usd_per_1000_pages": round(1000 * cost / pages, 2),
            "review_routing": routing,
            "validation": {
                "injected_errors_flagged": f"{sum(_flagged(r) for r in injected)}/{len(injected)}",
                "clean_docs_flagged": f"{sum(_flagged(r) for r in clean)}/{len(clean)}",
            },
        }
    return out


def _flagged(record: dict[str, Any]) -> bool:
    return any(i["code"] in ARITHMETIC_ISSUES for i in record["issues"])


def _round(value: float | None) -> float | None:
    return round(value) if value is not None else None
