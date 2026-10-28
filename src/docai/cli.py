"""Command line: `docai process invoice.pdf` or `docai serve`."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .pipeline import METHODS


def load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines into the environment without overriding variables already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="docai", description="Scanned invoice PDF -> validated JSON")
    sub = parser.add_subparsers(dest="command", required=True)
    proc = sub.add_parser("process", help="extract one PDF and print the result as JSON")
    proc.add_argument("pdf", type=Path)
    proc.add_argument("--method", choices=METHODS, default="hybrid")
    proc.add_argument("--lang", choices=("auto", "latin", "arabic"), default="auto")
    proc.add_argument("--full", action="store_true", help="include per-field confidence, issues and OCR text")
    serve = sub.add_parser("serve", help="run the review web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "process":
        from .pipeline import Pipeline

        result, _ = Pipeline().process(args.pdf, method=args.method, name=args.pdf.name, lang=args.lang)
        payload = (
            result.model_dump(mode="json")
            if args.full
            else {
                "invoice": result.invoice.model_dump(mode="json"),
                "needs_review": [f.name for f in result.fields if f.status.value == "needs_review"],
                "issues": [i.message or i.code for i in result.issues],
            }
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
