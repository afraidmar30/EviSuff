#!/usr/bin/env python3
"""Build paper-oriented Markdown/CSV tables from suite score files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--scores-dir", default="")
    parser.add_argument("--output-md", default="outputs/tables/deepseek_v4_flash_qwen3_paper/paper_tables.md")
    parser.add_argument("--output-csv", default="outputs/tables/deepseek_v4_flash_qwen3_paper/paper_tables.csv")
    parser.add_argument("--include-disabled", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def mean(rows: list[dict[str, Any]], field: str) -> float:
    vals = [float(row.get(field, 0.0) or 0.0) for row in rows]
    return statistics.fmean(vals) if vals else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    status_counts = Counter(str(row.get("status", "")) for row in rows)
    success = status_counts.get("success", 0)
    return {
        "n": n,
        "success_rate": success / n if n else 0.0,
        "answer_correctness": mean(rows, "answer_correctness"),
        "facet_coverage": mean(rows, "facet_coverage"),
        "evidence_support": mean(rows, "evidence_support"),
        "overall": mean(rows, "overall"),
        "search_calls": mean(rows, "search_calls"),
        "fetch_calls": mean(rows, "fetch_calls"),
        "citation_count": mean(rows, "citation_count"),
        "status_counts": dict(status_counts),
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"## {title}\n\nNo completed score files found.\n\n"
    headers = [
        "Method",
        "Model",
        "System",
        "n",
        "Success",
        "Overall",
        "Correct.",
        "Coverage",
        "Evidence",
        "Search",
        "Fetch",
        "Cites",
    ]
    lines = [f"## {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {method} | {model} | {system} | {n} | {success_rate:.4f} | "
            "{overall:.4f} | {answer_correctness:.4f} | {facet_coverage:.4f} | "
            "{evidence_support:.4f} | {search_calls:.2f} | {fetch_calls:.2f} | "
            "{citation_count:.2f} |".format(**row)
        )
    return "\n".join(lines) + "\n\n"


def main() -> None:
    args = parse_args()
    suite = load_yaml(args.suite)
    defaults = suite.get("defaults") or {}
    models = suite.get("models") or {}
    scores_dir_default = Path(args.scores_dir or defaults.get("scores_dir", "outputs/scores"))

    all_csv_rows: list[dict[str, Any]] = []
    md_parts = [f"# {suite.get('suite_name', 'experiment suite')} Paper Tables", ""]

    for group in suite.get("groups") or []:
        group_name = str(group.get("name") or "group")
        group_scores_dir = Path(group.get("scores_dir") or scores_dir_default)
        group_rows: list[dict[str, Any]] = []
        for run in group.get("runs") or []:
            if not args.include_disabled and not run.get("enabled", True):
                continue
            run_id = str(run.get("id"))
            rows = load_jsonl(group_scores_dir / f"{run_id}.scores.jsonl")
            summary = summarize_rows(rows)
            model_ref = str(run.get("model_ref") or "")
            model = models.get(model_ref, {}) if isinstance(models, dict) else {}
            out = {
                "group": group_name,
                "run_id": run_id,
                "method": str(run.get("method") or run_id),
                "model": str(model.get("label") or model_ref or run.get("model") or ""),
                "system": str(run.get("system") or ""),
                **summary,
            }
            group_rows.append(out)
            all_csv_rows.append(out)
        md_parts.append(markdown_table(group_name, group_rows))

    output_md = Path(args.output_md)
    output_csv = Path(args.output_csv)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md_parts), encoding="utf-8")

    fieldnames = [
        "group",
        "run_id",
        "method",
        "model",
        "system",
        "n",
        "success_rate",
        "overall",
        "answer_correctness",
        "facet_coverage",
        "evidence_support",
        "search_calls",
        "fetch_calls",
        "citation_count",
        "status_counts",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_csv_rows:
            row = dict(row)
            row["status_counts"] = json.dumps(row.get("status_counts", {}), ensure_ascii=False)
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(output_md)
    print(output_csv)


if __name__ == "__main__":
    main()
