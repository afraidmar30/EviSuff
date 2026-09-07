#!/usr/bin/env python3
"""Create balanced RAES JSONL shards for parallel trajectory collection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="raes-bench/raes-bench-v12_process/public_questions_1k.jsonl")
    parser.add_argument("--output-dir", default="raes-bench/raes-bench-v12_process/shards_1k")
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--prefix", default="public_questions_1k")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def row_sort_key(row: dict) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("category", "")),
        str(row.get("difficulty", "")),
        str(row.get("split", "")),
        str(row.get("task_type", "")),
        str(row.get("id", "")),
    )


def distribution(rows: list[dict], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field, "")) for row in rows))


def main() -> None:
    args = parse_args()
    rows = sorted(load_jsonl(Path(args.input)), key=row_sort_key)
    if args.shards <= 0:
        raise SystemExit("--shards must be positive")

    buckets: list[list[dict]] = [[] for _ in range(args.shards)]
    # Round-robin over a stratified ordering keeps category/difficulty/split
    # reasonably balanced without adding randomness.
    for index, row in enumerate(rows):
        buckets[index % args.shards].append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input": args.input,
        "output_dir": str(output_dir),
        "shards": args.shards,
        "total_rows": len(rows),
        "files": [],
    }
    for index, shard_rows in enumerate(buckets):
        path = output_dir / f"{args.prefix}_shard{index:02d}.jsonl"
        write_jsonl(path, shard_rows)
        manifest["files"].append(
            {
                "index": index,
                "path": str(path),
                "rows": len(shard_rows),
                "category": distribution(shard_rows, "category"),
                "difficulty": distribution(shard_rows, "difficulty"),
                "split": distribution(shard_rows, "split"),
            }
        )

    manifest_path = output_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
