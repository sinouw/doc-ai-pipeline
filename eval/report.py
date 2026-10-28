"""Render results/<run>/summary.json as a markdown report (results/<run>/report.md).

uv run python eval/report.py results/<run>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from docai.schema import REVIEW_FIELDS

ORDER = ("rules_raw", "rules", "ocr_llm", "vlm", "hybrid")
NAMES = {
    "rules_raw": "OCR + rules, no clean-up",
    "rules": "OCR + rules",
    "ocr_llm": "OCR text -> LLM",
    "vlm": "Vision LLM (image only)",
    "hybrid": "Hybrid (image + OCR text)",
}


def fmt(value: Any, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value}{suffix}"


def seconds(latency: dict[str, float]) -> str:
    return f"{latency['p50'] / 1000:.1f} / {latency['p95'] / 1000:.1f}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return [*lines, ""]


def invocations(run: dict[str, Any]) -> str:
    parts = []
    for inv in run.get("invocations", []):
        peak = f", peak RSS {inv['peak_rss_mb']} MB" if inv.get("peak_rss_mb") else ""
        stop = ", stopped on the daily quota" if inv.get("stopped_on_quota") else ""
        parts.append(f"`{inv.get('commit')}` {'/'.join(inv.get('methods', []))}{peak}{stop}")
    return "; ".join(parts) or "n/a"


def render(summary: dict[str, Any]) -> str:
    run = summary["run"]
    methods = {k: summary["methods"][k] for k in ORDER if k in summary["methods"]}
    docs = next(iter(methods.values()))["docs"]
    out = [
        f"# Evaluation run {run.get('started_at', '')}",
        "",
        f"- Split: `{run.get('split')}`. Methods are compared on the {docs} documents every method processed",
        f"  ({', '.join(summary.get('common_docs', []))})",
        f"- LLM: `{run.get('model')}` on Groq (free tier), temperature 0",
        f"- Machine: {run.get('machine')}, Python {run.get('python')}",
        f"- Invocations (commit, methods, peak memory): {invocations(run)}",
        f"- Deskew: mean absolute error {summary.get('deskew_mean_abs_error_deg')} degrees vs the generator's rotation",
        "",
        "## Headline",
        "",
    ]
    out += table(
        [
            "Method",
            "Field accuracy",
            "Doc exact match",
            "Latency p50 / p95 (s/page)",
            "Tokens in/out per page",
            "Cost per 1k pages",
        ],
        [
            [
                NAMES[k],
                fmt(m["field_accuracy"], "%"),
                fmt(m["exact_match"], "%"),
                seconds(m["latency_excl_rate_limit_ms_per_page"]),
                f"{m['tokens_per_page']['prompt']} / {m['tokens_per_page']['completion']}",
                f"${m['cost_usd_per_1000_pages']:.2f}",
            ]
            for k, m in methods.items()
        ],
    )
    out += [
        "Latency is wall clock per page on CPU (render, clean-up, OCR, extraction) minus time spent sleeping on HTTP 429",
        "back-offs; the raw wall clock and the back-off totals are in `summary.json`. Cost uses Groq's list price for the model",
        "(see `eval/run_eval.py`); the OCR + rules methods make no model call.",
        "",
        "## Field accuracy by language",
        "",
    ]
    langs = sorted(next(iter(methods.values()))["by"]["lang"])
    out += table(
        ["Method", *langs],
        [[NAMES[k], *(fmt(m["by"]["lang"][lang]["field_accuracy"], "%") for lang in langs)] for k, m in methods.items()],
    )
    out += ["Document exact match by language:", ""]
    out += table(
        ["Method", *langs],
        [[NAMES[k], *(fmt(m["by"]["lang"][lang]["exact_match"], "%") for lang in langs)] for k, m in methods.items()],
    )
    out += ["## Field accuracy by scan degradation", ""]
    sev = [s for s in ("light", "medium", "heavy") if s in next(iter(methods.values()))["by"]["severity"]]
    out += table(
        ["Method", *sev],
        [[NAMES[k], *(fmt(m["by"]["severity"][s]["field_accuracy"], "%") for s in sev)] for k, m in methods.items()],
    )
    out += ["## Accuracy per field", ""]
    out += table(
        ["Field", *(NAMES[k] for k in methods)],
        [[f, *(fmt(m["field_accuracy_by_field"][f], "%") for m in methods.values())] for f in REVIEW_FIELDS]
        + [["line items incl. descriptions", *(fmt(m["line_items_with_descriptions"], "%") for m in methods.values())]],
    )
    out += [
        "## Human review routing",
        "",
        "A field goes to the review queue when its confidence is below the threshold or validation blames it.",
        "`errors caught` is the share of wrong fields that landed in the queue; `auto-accepted accuracy` is the",
        "accuracy of the fields that skipped it.",
        "",
    ]
    for k, m in methods.items():
        out += [f"**{NAMES[k]}**", ""]
        out += table(
            ["Threshold", "Fields sent to review", "Auto-accepted accuracy", "Errors caught", "Errors missed"],
            [
                [
                    t,
                    fmt(r["review_rate"], "%"),
                    fmt(r["auto_accepted_accuracy"], "%"),
                    fmt(r["errors_caught"], "%"),
                    str(r["errors_missed"]),
                ]
                for t, r in m["review_routing"].items()
            ],
        )
    out += [
        "## Validation",
        "",
        "Some invoices print a total that does not equal subtotal + VAT. The table shows how often the arithmetic checks",
        "flag those documents, and how often they flag documents whose printed numbers are consistent (a false alarm,",
        "caused by an extraction error).",
        "",
    ]
    out += table(
        ["Method", "Inconsistent invoices flagged", "Consistent invoices flagged", "LLM calls", "429 retries", "Errors"],
        [
            [
                NAMES[k],
                m["validation"]["injected_errors_flagged"],
                m["validation"]["clean_docs_flagged"],
                str(m["llm_calls"]),
                str(m["rate_limit_retries"]),
                str(m["errors"]),
            ]
            for k, m in methods.items()
        ],
    )
    every = summary.get("methods_all_docs", {})
    extra = {k: m for k, m in every.items() if m["docs"] > docs}
    if extra:
        out += [
            "## Methods run on more documents",
            "",
            "The OCR-only methods cost nothing to run, so they also ran on the full split.",
            "",
        ]
        out += table(
            [
                "Method",
                "Docs",
                "Field accuracy",
                "Doc exact match",
                *(f"{lang} field acc." for lang in langs),
                "Latency p50 / p95 (s/page)",
            ],
            [
                [
                    NAMES[k],
                    str(m["docs"]),
                    fmt(m["field_accuracy"], "%"),
                    fmt(m["exact_match"], "%"),
                    *(fmt(m["by"]["lang"].get(lang, {}).get("field_accuracy"), "%") for lang in langs),
                    seconds(m["latency_excl_rate_limit_ms_per_page"]),
                ]
                for k, m in sorted(extra.items(), key=lambda kv: ORDER.index(kv[0]))
            ],
        )
    return "\n".join(out)


def main() -> None:
    run_dir = Path(sys.argv[1])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    (run_dir / "report.md").write_text(render(summary) + "\n", encoding="utf-8")
    print(run_dir / "report.md")


if __name__ == "__main__":
    main()
