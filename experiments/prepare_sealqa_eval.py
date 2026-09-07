#!/usr/bin/env python3
"""Prepare SealQA parquet splits as RAES-like JSONL for SearchClaw eval."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


SPLIT_FILES = {
    "seal0": "seal-0.parquet",
    "seal-0": "seal-0.parquet",
    "seal_hard": "seal-hard.parquet",
    "seal-hard": "seal-hard.parquet",
    "longseal": "longseal.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/sealqa")
    parser.add_argument("--split", default="seal0", choices=sorted(SPLIT_FILES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument(
        "--no-effective-year-note",
        action="store_true",
        help="Do not append the SealQA effective_year note to the question.",
    )
    return parser.parse_args()


def as_plain(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple)):
        return [as_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return as_plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): as_plain(v) for k, v in value.items()}
    return value


def stratified_sample(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if not limit or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("topic") or ""),
            str(row.get("freshness") or ""),
            str(row.get("search_results") or ""),
        )
        groups.setdefault(key, []).append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    selected: list[dict[str, Any]] = []
    keys = sorted(groups, key=lambda key: (-len(groups[key]), key))
    while len(selected) < limit and keys:
        next_keys = []
        for key in keys:
            bucket = groups[key]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def build_question(row: dict[str, Any], include_effective_year: bool) -> str:
    question = str(row.get("question") or "").strip()
    if include_effective_year and row.get("effective_year"):
        question += (
            f"\n\nTarget temporal context: answer according to public evidence relevant to "
            f"{row['effective_year']}. If current evidence conflicts with that target, state the uncertainty."
        )
    return question


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.data_dir) / SPLIT_FILES[args.split]
    df = pd.read_parquet(parquet_path)
    rows = [{key: as_plain(value) for key, value in row.items()} for row in df.to_dict("records")]
    rows = stratified_sample(rows, args.limit, args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, 1):
            answer = row.get("answer")
            if isinstance(answer, list):
                answer_text = "; ".join(str(item) for item in answer)
            else:
                answer_text = str(answer or "").strip()
            out = {
                "id": f"sealqa-{args.split}-{idx:03d}",
                "question": build_question(row, not args.no_effective_year_note),
                "category": str(row.get("topic") or "sealqa"),
                "domain": "sealqa",
                "task_type": "external_realtime_search_qa",
                "difficulty": "hard" if "hard" in args.split else "",
                "split": args.split,
                "answerability": "answerable" if answer_text else "",
                "gold_answer": answer_text,
                "gold_sources": [{"url": str(url)} for url in (row.get("urls") or []) if url],
                "source_dataset": "sealqa",
                "source_record": row,
            }
            f.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(output)


if __name__ == "__main__":
    main()
