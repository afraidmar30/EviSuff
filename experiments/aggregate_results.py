#!/usr/bin/env python3
"""Aggregate RAES score JSONL files into JSON and Markdown tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.raes_eval.metrics import aggregate_scores, load_jsonl, markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--output-json", default="outputs/tables/raes_summary.json")
    parser.add_argument("--output-md", default="outputs/tables/raes_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.scores:
        rows.extend(load_jsonl(path))
    summary = aggregate_scores(rows)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(markdown_table(summary), encoding="utf-8")
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
