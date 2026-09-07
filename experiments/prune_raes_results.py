#!/usr/bin/env python3
"""Prune low-quality RAES result rows so resume will rerun them."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.answer_quality import looks_like_procedural_fragment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="*", default=[], help="Explicit results.jsonl files to prune.")
    parser.add_argument("--run-glob", default="", help="Glob under --output-dir matching run directories.")
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--min-citations", type=int, default=3)
    parser.add_argument("--min-answer-chars", type=int, default=800)
    parser.add_argument(
        "--drop-gate-decisions",
        default="continue_search",
        help="Comma-separated RAES gate decisions to prune from success rows.",
    )
    parser.add_argument("--apply", action="store_true", help="Rewrite files. Without this, only report.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        row, idx = decoder.raw_decode(text, idx)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def gate_decision(row: dict[str, Any]) -> str:
    gate = row.get("raes_gate")
    if isinstance(gate, dict):
        return str(gate.get("decision") or "")
    return str(row.get("gate_decision") or "")


def final_answer(row: dict[str, Any]) -> str:
    return str(row.get("final_answer") or row.get("answer") or "").strip()


def reject_reasons(row: dict[str, Any], args: argparse.Namespace, drop_gates: set[str]) -> list[str]:
    reasons: list[str] = []
    status = row.get("status", "success")
    answer = final_answer(row)
    citations = row.get("citations") or []
    decision = gate_decision(row)

    if status != "success":
        reasons.append("status_not_success")
    if not answer:
        reasons.append("empty_final_answer")
    if len(answer) < args.min_answer_chars:
        reasons.append("final_answer_too_short")
    if looks_like_procedural_fragment(answer):
        reasons.append("procedural_fragment_final_answer")
    if len(citations) < args.min_citations:
        reasons.append("too_few_citations")
    if decision in drop_gates:
        reasons.append(f"gate_{decision}")
    return reasons


def result_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(item) for item in args.results]
    if args.run_glob:
        base = Path(args.output_dir)
        paths.extend(path / "results.jsonl" for path in sorted(base.glob(args.run_glob)) if path.is_dir())
    return [path for path in paths if path.exists()]


def main() -> None:
    args = parse_args()
    paths = result_paths(args)
    if not paths:
        raise SystemExit("No results.jsonl files found.")
    drop_gates = {item.strip() for item in args.drop_gate_decisions.split(",") if item.strip()}
    total_kept = 0
    total_dropped = 0
    all_reasons: Counter[str] = Counter()

    for path in paths:
        rows = read_rows(path)
        kept: list[dict[str, Any]] = []
        dropped: list[tuple[dict[str, Any], list[str]]] = []
        for row in rows:
            reasons = reject_reasons(row, args, drop_gates)
            if reasons:
                dropped.append((row, reasons))
                all_reasons.update(reasons)
            else:
                kept.append(row)

        total_kept += len(kept)
        total_dropped += len(dropped)
        print(f"{path}: kept={len(kept)} dropped={len(dropped)} total={len(rows)}")
        for row, reasons in dropped[:20]:
            print(f"  DROP {row.get('sample_id')} reasons={','.join(reasons)}")
        if len(dropped) > 20:
            print(f"  ... {len(dropped) - 20} more")

        if args.apply and dropped:
            backup = path.with_suffix(path.suffix + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.copy2(path, backup)
            payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in kept)
            path.write_text(payload, encoding="utf-8")
            print(f"  wrote {path}; backup={backup}")

    print(f"SUMMARY kept={total_kept} dropped={total_dropped} files={len(paths)}")
    print("REASONS", dict(all_reasons))
    if not args.apply:
        print("Dry run only. Add --apply to rewrite results.jsonl files.")


if __name__ == "__main__":
    main()
