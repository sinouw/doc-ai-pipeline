"""Run every extraction method over a labelled split and write raw records plus a summary.

    uv run python eval/run_eval.py --split data/test --out results/<run-id>
    uv run python eval/run_eval.py --out results/<run-id> --resume      # continue an interrupted run
    uv run python eval/run_eval.py --out results/<run-id> --summarize   # recompute summary only

Methods:
    rules      OCR (with clean-up) + keyword rules, no model call
    rules_raw  same without denoise/deskew (preprocessing ablation)
    ocr_llm    OCR text -> LLM
    vlm        page image -> vision LLM
    hybrid     page image + OCR text -> vision LLM
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docai.evaluation import score_record, summarize
from docai.llm import GroqClient, QuotaExhaustedError
from docai.pipeline import Pipeline, PreparedDocument

ALL_METHODS = ("rules", "rules_raw", "ocr_llm", "vlm", "hybrid")
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
# Stated next to every latency number: the OCR runs on this CPU.
MACHINE = "MacBook Pro 2019, Intel Core i7-9750H (6 cores / 12 threads), 16 GB RAM, CPU only"
# USD per 1M tokens (input, output) from https://console.groq.com/docs/models, read 2026-10-28.
PRICES = {
    "qwen/qwen3.8-27b": (0.80, 4.00),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record_for(doc_meta: dict[str, Any], method: str, doc: PreparedDocument, pipeline: Pipeline) -> dict[str, Any]:
    label = doc_meta["label"]
    meta = doc_meta["meta"]
    base = {
        "id": doc_meta["id"],
        "method": method,
        "lang": meta["lang"],
        "severity": meta["severity"],
        "injected_error": meta["injected_error"],
        "skew_true": meta.get("skew_degrees"),
        "skew_estimated": doc.skew_degrees[0] if doc.skew_degrees else None,
        "label": label,
    }
    extract_method = "rules" if method == "rules_raw" else method
    try:
        result = pipeline.extract(doc, extract_method)  # type: ignore[arg-type]
    except QuotaExhaustedError:
        raise  # stop the run without recording a failure; --resume picks it up later
    except Exception as exc:  # keep going: a failed document is recorded and counted as wrong
        logging.exception("%s %s failed", doc_meta["id"], method)
        return {
            **base,
            "error": str(exc),
            "prediction": {},
            "correct": dict.fromkeys(label, False) | {"line_items": False},
            "fields": [],
            "issues": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "retries": 0, "wait_ms": 0.0},
            "timings_ms": {"total": 0, "extract": 0},
            "pages": len(doc.pages),
        }
    prediction = result.invoice.model_dump(mode="json")
    return {
        **base,
        "error": None,
        "prediction": prediction,
        "correct": score_record(prediction, label),
        "fields": [f.model_dump(mode="json") for f in result.fields],
        "issues": [i.model_dump() for i in result.issues],
        "usage": result.usage.model_dump(),
        "timings_ms": result.timings_ms,
        "pages": result.pages,
    }


def write_summary(out: Path, model: str) -> dict[str, Any]:
    records = load_records(out / "records.jsonl")
    prices = {m: PRICES.get(model, (0.0, 0.0)) for m in ("ocr_llm", "vlm", "hybrid")}
    # Headline: only documents that every method processed, so the methods are compared on the same pages.
    methods_per_doc: dict[str, set[str]] = {}
    for r in records:
        methods_per_doc.setdefault(r["id"], set()).add(r["method"])
    all_methods = set().union(*methods_per_doc.values()) if methods_per_doc else set()
    common = {doc for doc, ms in methods_per_doc.items() if ms == all_methods}
    summary = summarize([r for r in records if r["id"] in common], prices, THRESHOLDS)
    every_doc = summarize(records, prices, THRESHOLDS)
    skew = [(r["skew_true"], r["skew_estimated"]) for r in records if r["method"] == "rules" and r["skew_true"] is not None]
    run_info = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
    payload = {
        "run": run_info,
        "deskew_mean_abs_error_deg": round(sum(abs(t + e) for t, e in skew) / len(skew), 3) if skew else None,
        "common_docs": sorted(common),
        "methods": summary,
        "methods_all_docs": every_doc,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, default=Path("data/test"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-language", type=int, default=0, help="only the first N documents of each language")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true", help="only recompute summary.json from records.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    records_path = args.out / "records.jsonl"
    llm_methods = {"ocr_llm", "vlm", "hybrid"} & set(methods)
    client = GroqClient() if llm_methods and not args.summarize else None
    model = (
        client.model
        if client
        else json.loads((args.out / "run.json").read_text()).get("model", "")
        if (args.out / "run.json").exists()
        else ""
    )

    if args.summarize:
        print(json.dumps(write_summary(args.out, model)["methods"], indent=2)[:4000])
        return

    if not args.resume and records_path.exists():
        sys.exit(f"{records_path} exists; pass --resume to continue it or choose another --out")
    run_file = args.out / "run.json"
    run_info = json.loads(run_file.read_text()) if args.resume and run_file.exists() else {}
    invocation = {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "methods": methods,
        "per_language": args.per_language or None,
        "commit": git_commit(),
    }
    run_info.setdefault("started_at", invocation["started_at"])
    run_info.update(
        {
            "split": str(args.split),
            "model": model or run_info.get("model", ""),
            "machine": run_info.get("machine", MACHINE),
            "python": platform.python_version(),
        }
    )
    run_info.setdefault("invocations", []).append(invocation)
    run_file.write_text(json.dumps(run_info, indent=2) + "\n")
    done = {(r["id"], r["method"]) for r in load_records(records_path)}
    pipeline = Pipeline(client=client)
    raw_pipeline = Pipeline(ocr=pipeline.ocr, client=client, denoise=False, deskew=False)
    labels = sorted(args.split.glob("*.json"))
    labels = [p for p in labels if p.name != "manifest.json"]
    if args.per_language:
        seen: dict[str, int] = {}
        kept = []
        for p in labels:
            lang = p.stem.split("-")[0]
            seen[lang] = seen.get(lang, 0) + 1
            if seen[lang] <= args.per_language:
                kept.append(p)
        labels = kept
    if args.limit:
        labels = labels[: args.limit]

    quota_hit = None
    with records_path.open("a", encoding="utf-8") as sink:
        for i, label_path in enumerate(labels, 1):
            if quota_hit:
                break
            doc_meta = json.loads(label_path.read_text(encoding="utf-8"))
            todo = [m for m in methods if (doc_meta["id"], m) not in done]
            if not todo:
                continue
            pdf = label_path.with_suffix(".pdf")
            t0 = time.perf_counter()
            prepared = pipeline.prepare(pdf, name=doc_meta["id"]) if set(todo) - {"rules_raw"} else None
            raw = raw_pipeline.prepare(pdf, name=doc_meta["id"]) if "rules_raw" in todo else None
            for method in todo:
                doc = raw if method == "rules_raw" else prepared
                assert doc is not None
                try:
                    record = record_for(doc_meta, method, doc, raw_pipeline if method == "rules_raw" else pipeline)
                except QuotaExhaustedError as exc:
                    quota_hit = str(exc)
                    break
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                score = sum(record["correct"].values())
                logging.info(
                    "[%d/%d] %s %-9s %2d/12 fields%s",
                    i,
                    len(labels),
                    doc_meta["id"],
                    method,
                    score,
                    f" ERROR {record['error'][:80]}" if record["error"] else "",
                )
            logging.info("%s done in %.1fs", doc_meta["id"], time.perf_counter() - t0)

    # ru_maxrss is bytes on macOS, kilobytes on Linux
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)
    invocation["peak_rss_mb"] = round(peak)
    invocation["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    invocation["stopped_on_quota"] = quota_hit
    run_file.write_text(json.dumps(run_info, indent=2) + "\n")
    payload = write_summary(args.out, model)
    if quota_hit:
        logging.warning("stopped: %s. Run again with --resume once the quota has recovered.", quota_hit)
    for method, m in payload["methods"].items():
        print(
            f"{method:10s} field acc {m['field_accuracy']}%  exact {m['exact_match']}%  "
            f"p50 {m['latency_ms_per_page']['p50']} ms/page  ${m['cost_usd_per_1000_pages']}/1k pages"
        )


if __name__ == "__main__":
    main()
