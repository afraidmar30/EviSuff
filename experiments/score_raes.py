#!/usr/bin/env python3
"""Score a RAES result JSONL with the baseline heuristic judge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.raes_eval.judge import score_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="raes-bench/raes-bench-v12_process/gold.jsonl")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = score_file(args.gold, args.results)
    output = Path(args.output) if args.output else Path(args.results).with_name("scores.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in scores:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(output)


if __name__ == "__main__":
    main()
