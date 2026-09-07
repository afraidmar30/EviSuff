#!/usr/bin/env python3
"""Score external QA runs with simple answer matching and tool statistics."""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="RAES-like external QA JSONL.")
    parser.add_argument("--results", required=True, help="SearchClaw results.jsonl.")
    parser.add_argument("--output-dir", default="", help="Defaults to the results directory.")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\\S+", " ", text)
    text = re.sub(r"\\[[^\\]]*\\]\\([^)]*\\)", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\\b(a|an|the)\\b", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def answer_matches(gold_answer: str, final_answer: str) -> dict[str, bool]:
    gold = normalize(gold_answer)
    pred = normalize(final_answer)
    exact = bool(gold) and gold == pred
    contains = bool(gold) and gold in pred
    token_recall = False
    if gold and pred:
        gold_tokens = [token for token in gold.split() if token]
        if gold_tokens:
            covered = sum(1 for token in gold_tokens if token in pred.split())
            token_recall = covered / len(gold_tokens) >= 0.8
    return {
        "exact_match": exact,
        "contains_answer": contains,
        "soft_match": exact or contains or token_recall,
    }


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0) or 0) for row in rows]
    return mean(values) if values else 0.0


def main() -> None:
    args = parse_args()
    gold_rows = {str(row["id"]): row for row in load_jsonl(args.gold)}
    result_rows = load_jsonl(args.results)
    scored = []
    for result in result_rows:
        sample_id = str(result.get("sample_id") or result.get("id") or "")
        gold = gold_rows.get(sample_id, {})
        gold_answer = str(gold.get("gold_answer") or "")
        final_answer = str(result.get("final_answer") or "")
        matches = answer_matches(gold_answer, final_answer)
        tool_stats = result.get("tool_stats") or {}
        scored.append({
            "sample_id": sample_id,
            "status": result.get("status", ""),
            "gold_answer": gold_answer,
            "final_answer": final_answer,
            **matches,
            "search_calls": tool_stats.get("search_calls", 0),
            "fetch_calls": tool_stats.get("fetch_calls", 0),
            "total_tool_calls": tool_stats.get("total_tool_calls", 0),
            "citation_count": len(result.get("citations") or []),
            "wall_time_sec": (result.get("cost_stats") or {}).get("wall_time_sec", 0),
        })

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.results).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "external_qa_scores.jsonl"
    compat_scores_path = out_dir / "scores.jsonl"
    score_lines = []
    for row in scored:
        score_lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with scores_path.open("w", encoding="utf-8") as f:
        f.writelines(score_lines)
    with compat_scores_path.open("w", encoding="utf-8") as f:
        f.writelines(score_lines)

    n = len(scored)
    success = sum(1 for row in scored if row["status"] == "success")
    exact = sum(1 for row in scored if row["exact_match"])
    contains = sum(1 for row in scored if row["contains_answer"])
    soft = sum(1 for row in scored if row["soft_match"])
    summary = {
        "n": n,
        "success_rate": success / n if n else 0,
        "exact_match": exact / n if n else 0,
        "contains_answer": contains / n if n else 0,
        "soft_match": soft / n if n else 0,
        "avg_search_calls": avg(scored, "search_calls"),
        "avg_fetch_calls": avg(scored, "fetch_calls"),
        "avg_total_tool_calls": avg(scored, "total_tool_calls"),
        "avg_citation_count": avg(scored, "citation_count"),
        "avg_wall_time_sec": avg(scored, "wall_time_sec"),
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    summary_path = out_dir / "external_qa_summary.json"
    compat_summary_path = out_dir / "summary.json"
    summary_path.write_text(summary_text, encoding="utf-8")
    compat_summary_path.write_text(summary_text, encoding="utf-8")

    md = "\n".join([
        "# External QA Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| N | {n} |",
        f"| Success | {success}/{n} ({pct(summary['success_rate'])}) |",
        f"| Exact match | {exact}/{n} ({pct(summary['exact_match'])}) |",
        f"| Contains answer | {contains}/{n} ({pct(summary['contains_answer'])}) |",
        f"| Soft match | {soft}/{n} ({pct(summary['soft_match'])}) |",
        f"| Avg search calls | {summary['avg_search_calls']:.2f} |",
        f"| Avg fetch calls | {summary['avg_fetch_calls']:.2f} |",
        f"| Avg total tool calls | {summary['avg_total_tool_calls']:.2f} |",
        f"| Avg citations | {summary['avg_citation_count']:.2f} |",
        f"| Avg wall time sec | {summary['avg_wall_time_sec']:.1f} |",
        "",
        "## Files",
        "",
        f"- Scores: `{compat_scores_path}`",
        f"- Summary JSON: `{compat_summary_path}`",
        "",
    ])
    md_path = out_dir / "external_qa_summary.md"
    compat_md_path = out_dir / "summary.md"
    md_path.write_text(md, encoding="utf-8")
    compat_md_path.write_text(md, encoding="utf-8")
    print(compat_md_path)


if __name__ == "__main__":
    main()
