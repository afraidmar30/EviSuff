#!/usr/bin/env python3
"""Create an annotation template for RAES failure analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ERROR_TYPES = [
    "query_error",
    "retrieval_miss",
    "fetch_failure",
    "evidence_misread",
    "incomplete_facets",
    "citation_mismatch",
    "premature_stop",
    "over_search",
    "teacher_noise",
    "judge_noise",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--results", nargs="*", default=[])
    parser.add_argument("--output", default="outputs/analysis/error_analysis_template.csv")
    parser.add_argument("--lowest", type=int, default=20)
    parser.add_argument("--run-label", default="")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def result_index(paths: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            key = (str(row.get("run_id", "")), str(row.get("sample_id", "")))
            index[key] = row
    return index


def main() -> None:
    args = parse_args()
    result_rows = result_index(args.results)
    scored: list[dict[str, Any]] = []
    for path in args.scores:
        for row in load_jsonl(path):
            row = dict(row)
            row["_score_path"] = path
            scored.append(row)
    scored.sort(key=lambda row: float(row.get("overall", 0.0) or 0.0))
    selected = scored[: args.lowest]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_label",
        "run_id",
        "sample_id",
        "agent",
        "status",
        "overall",
        "answer_correctness",
        "facet_coverage",
        "evidence_support",
        "search_calls",
        "fetch_calls",
        "citation_count",
        "question",
        "answer_excerpt",
        "primary_error_type",
        "secondary_error_type",
        "human_correctness_1_5",
        "human_evidence_1_5",
        "human_citation_1_5",
        "human_completeness_1_5",
        "harmful_hallucination_yes_no",
        "notes",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            key = (str(row.get("run_id", "")), str(row.get("sample_id", "")))
            result = result_rows.get(key, {})
            answer = str(result.get("final_answer", ""))
            writer.writerow(
                {
                    "run_label": args.run_label,
                    "run_id": row.get("run_id", ""),
                    "sample_id": row.get("sample_id", ""),
                    "agent": row.get("agent", ""),
                    "status": row.get("status", ""),
                    "overall": row.get("overall", ""),
                    "answer_correctness": row.get("answer_correctness", ""),
                    "facet_coverage": row.get("facet_coverage", ""),
                    "evidence_support": row.get("evidence_support", ""),
                    "search_calls": row.get("search_calls", ""),
                    "fetch_calls": row.get("fetch_calls", ""),
                    "citation_count": row.get("citation_count", ""),
                    "question": result.get("question", ""),
                    "answer_excerpt": answer[:800],
                    "primary_error_type": "",
                    "secondary_error_type": "",
                    "human_correctness_1_5": "",
                    "human_evidence_1_5": "",
                    "human_citation_1_5": "",
                    "human_completeness_1_5": "",
                    "harmful_hallucination_yes_no": "",
                    "notes": f"Allowed error types: {', '.join(ERROR_TYPES)}",
                }
            )
    print(output)


if __name__ == "__main__":
    main()
