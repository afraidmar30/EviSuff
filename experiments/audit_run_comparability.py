#!/usr/bin/env python3
"""Audit whether result files are evaluated on the same task IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=RESULTS_JSONL")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def parse_runs(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got {value}")
        label, path = value.split("=", 1)
        if label in out:
            raise ValueError(f"Duplicate label: {label}")
        out[label] = Path(path)
    return out


def load(path: Path) -> tuple[set[str], dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    ids = {str(row.get("sample_id") or row.get("id") or "") for row in rows}
    ids.discard("")
    metadata = {
        "rows": len(rows),
        "unique_task_ids": len(ids),
        "run_ids": sorted({str(row.get("run_id") or "") for row in rows}),
        "agents": sorted({str(row.get("agent") or "") for row in rows}),
        "models": sorted({str(row.get("model") or "") for row in rows}),
    }
    return ids, metadata


def main() -> None:
    args = parse_args()
    paths = parse_runs(args.runs)
    loaded = {label: load(path) for label, path in paths.items()}
    all_sets = [item[0] for item in loaded.values()]
    common = set.intersection(*all_sets) if all_sets else set()
    union = set.union(*all_sets) if all_sets else set()
    pairs: dict[str, Any] = {}
    labels = list(loaded)
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            left_ids, right_ids = loaded[left][0], loaded[right][0]
            overlap = left_ids & right_ids
            pair_union = left_ids | right_ids
            pairs[f"{left} vs {right}"] = {
                "overlap": len(overlap),
                "left_n": len(left_ids),
                "right_n": len(right_ids),
                "jaccard": len(overlap) / len(pair_union) if pair_union else None,
                "same_task_set": left_ids == right_ids,
            }
    report = {
        "runs": {label: {"path": str(paths[label]), **loaded[label][1]} for label in labels},
        "all_run_intersection": len(common),
        "all_run_union": len(union),
        "all_runs_same_task_set": bool(all_sets) and all(ids == all_sets[0] for ids in all_sets[1:]),
        "pairs": pairs,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Run comparability audit", "",
             f"All runs share the same task set: **{report['all_runs_same_task_set']}**", "",
             f"All-run intersection / union: {len(common)} / {len(union)}", "",
             "| Pair | Left N | Right N | Overlap | Jaccard | Same set |",
             "|---|---:|---:|---:|---:|---:|"]
    for pair, values in pairs.items():
        lines.append(
            f"| {pair} | {values['left_n']} | {values['right_n']} | {values['overlap']} | "
            f"{values['jaccard']:.3f} | {values['same_task_set']} |"
        )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
