#!/usr/bin/env python3
"""Create a question-disjoint answer-SFT corpus.

The historical answer-SFT corpus was split by rows and included EviSuff-BoundaryBench
test/stress questions.  This utility combines the original train/validation
files, removes every row whose user prompt contains a held-out question, then
re-splits by normalized user question so one question cannot cross splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--exclude-question-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            if isinstance(row, dict):
                yield line_no, row


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_user_text(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            return normalize_text(message.get("content"))
    return ""


def collect_exclusions(paths: list[str]) -> dict[str, str]:
    exclusions: dict[str, str] = {}
    for raw_path in paths:
        for _, row in load_jsonl(Path(raw_path)):
            sample_id = str(row.get("id") or row.get("sample_id") or "")
            question = normalize_text(row.get("question"))
            if sample_id and question:
                exclusions[sample_id] = question
    return exclusions


def split_name(question: str, seed: int, val_percent: int) -> str:
    digest = hashlib.sha256(f"{seed}:{question}".encode("utf-8")).hexdigest()
    return "val" if int(digest[:8], 16) % 100 < val_percent else "train"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    if not 1 <= args.val_percent <= 50:
        raise SystemExit("--val-percent must be between 1 and 50")
    exclusions = collect_exclusions(args.exclude_question_files)
    kept: list[tuple[str, dict[str, Any]]] = []
    excluded_rows = 0
    excluded_ids: Counter[str] = Counter()
    missing_user = 0
    input_rows = 0
    for raw_path in args.inputs:
        for _, row in load_jsonl(Path(raw_path)):
            input_rows += 1
            user_text = first_user_text(row)
            if not user_text:
                missing_user += 1
                continue
            matched_id = next(
                (
                    sample_id
                    for sample_id, question in exclusions.items()
                    if question == user_text or question in user_text
                ),
                None,
            )
            if matched_id:
                excluded_rows += 1
                excluded_ids[matched_id] += 1
                continue
            kept.append((user_text, {"messages": row.get("messages") or []}))

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    split_questions: dict[str, set[str]] = {"train": set(), "val": set()}
    for question, row in kept:
        split = split_name(question, args.seed, args.val_percent)
        split_rows[split].append(row)
        split_questions[split].add(question)
    overlap = split_questions["train"].intersection(split_questions["val"])
    if overlap:
        raise RuntimeError(f"Question-level split overlap: {len(overlap)}")

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", split_rows["train"])
    write_jsonl(output_dir / "val.jsonl", split_rows["val"])
    report = {
        "design": "answer-SFT with question-level held-out filtering and splitting",
        "inputs": args.inputs,
        "exclude_question_files": args.exclude_question_files,
        "heldout_question_ids": len(exclusions),
        "input_rows": input_rows,
        "excluded_rows": excluded_rows,
        "excluded_unique_question_ids": len(excluded_ids),
        "excluded_rows_by_question_id": dict(sorted(excluded_ids.items())),
        "rows_without_user_message_dropped": missing_user,
        "remaining_rows": len(kept),
        "train_rows": len(split_rows["train"]),
        "val_rows": len(split_rows["val"]),
        "train_unique_questions": len(split_questions["train"]),
        "val_unique_questions": len(split_questions["val"]),
        "train_val_question_overlap": len(overlap),
        "val_percent": args.val_percent,
        "seed": args.seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
