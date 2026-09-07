#!/usr/bin/env python3
"""Select SFT rows using gold scores when available and no-gold quality signals otherwise."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", required=True, help="Input sft_messages.jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scores", nargs="*", default=[], help="Optional scores.jsonl files for gold-backed rows.")
    parser.add_argument("--min-overall", type=float, default=0.0)
    parser.add_argument("--min-final-answer-chars", type=int, default=200)
    parser.add_argument("--min-citations", type=int, default=2)
    parser.add_argument("--min-evidence-results", type=int, default=1)
    parser.add_argument("--max-tool-error-ratio", type=float, default=0.3)
    parser.add_argument("--max-turns", type=int, default=0, help="0 disables the turn cap.")
    parser.add_argument("--allow-gate-failed", action="store_true")
    parser.add_argument("--best-of-sample", action="store_true", help="Keep only the best row per sample_id.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_scores(paths: list[str]) -> dict[tuple[str, str], dict]:
    scores: dict[tuple[str, str], dict] = {}
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates = sorted(path.glob("**/scores.jsonl"))
        else:
            candidates = [path]
        for candidate in candidates:
            if not candidate.exists():
                continue
            for row in load_jsonl(candidate):
                scores[(str(row.get("run_id", "")), str(row.get("sample_id", "")))] = row
    return scores


def citation_domains(row: dict) -> set[str]:
    domains: set[str] = set()
    for citation in row.get("citations", []) or []:
        url = str(citation.get("url", "") or "")
        if not url:
            continue
        domains.add(urlparse(url).netloc.replace("www.", ""))
    return {domain for domain in domains if domain}


def row_score(row: dict) -> tuple[float, float, int, int]:
    metadata = row.get("metadata", {}) or {}
    gold_score = metadata.get("raes_heuristic_score") or {}
    quality = row.get("quality_signals", {}) or {}
    overall = float(gold_score.get("overall", -1.0) if gold_score else -1.0)
    evidence = float(gold_score.get("evidence_support", -1.0) if gold_score else -1.0)
    citations = int(quality.get("citation_count", 0) or 0)
    turns = int((metadata.get("tool_stats", {}) or {}).get("turns", 9999) or 9999)
    return (overall, evidence, citations, -turns)


def rejection_reasons(row: dict, args: argparse.Namespace) -> list[str]:
    quality = row.get("quality_signals", {}) or {}
    metadata = row.get("metadata", {}) or {}
    score = metadata.get("raes_heuristic_score") or {}
    reasons: list[str] = []
    if quality.get("status") != "success" and metadata.get("status") != "success":
        reasons.append("not_success")
    if int(quality.get("final_answer_chars", 0) or 0) < args.min_final_answer_chars:
        reasons.append("short_final_answer")
    if int(quality.get("citation_count", 0) or 0) < args.min_citations:
        reasons.append("too_few_citations")
    if int(quality.get("evidence_result_count", 0) or 0) < args.min_evidence_results:
        reasons.append("no_source_evidence")
    if float(quality.get("tool_error_ratio", 0.0) or 0.0) > args.max_tool_error_ratio:
        reasons.append("tool_errors_dominate")
    if not args.allow_gate_failed and quality.get("gate_passed") is False and not quality.get("forced_by_gate_budget"):
        reasons.append("gate_failed")
    if args.max_turns:
        turns = int((metadata.get("tool_stats", {}) or {}).get("turns", 0) or 0)
        if turns > args.max_turns:
            reasons.append("too_many_turns")
    if score and float(score.get("overall", -1.0)) < args.min_overall:
        reasons.append("low_gold_heuristic_score")
    return reasons


def attach_scores(rows: list[dict], scores: dict[tuple[str, str], dict]) -> None:
    for row in rows:
        metadata = row.setdefault("metadata", {})
        key = (str(metadata.get("run_id", "")), str(metadata.get("sample_id", "")))
        score = scores.get(key)
        if score:
            metadata["raes_heuristic_score"] = {
                key: score[key]
                for key in (
                    "answer_correctness",
                    "facet_coverage",
                    "evidence_support",
                    "overall",
                    "search_calls",
                    "fetch_calls",
                    "citation_count",
                )
                if key in score
            }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.sft))
    attach_scores(rows, load_scores(args.scores))

    accepted: list[dict] = []
    rejected: list[dict] = []
    reason_counts: Counter = Counter()
    for row in rows:
        reasons = rejection_reasons(row, args)
        if reasons:
            for reason in reasons:
                reason_counts[reason] += 1
            rejected.append(
                {
                    "id": row.get("id"),
                    "sample_id": (row.get("metadata", {}) or {}).get("sample_id"),
                    "run_id": (row.get("metadata", {}) or {}).get("run_id"),
                    "reasons": reasons,
                    "quality_signals": row.get("quality_signals", {}),
                    "raes_heuristic_score": (row.get("metadata", {}) or {}).get("raes_heuristic_score", {}),
                }
            )
            continue
        accepted.append(row)

    accepted_before_best_of = len(accepted)
    if args.best_of_sample:
        best: dict[str, dict] = {}
        for row in accepted:
            sample_id = str((row.get("metadata", {}) or {}).get("sample_id") or row.get("id"))
            current = best.get(sample_id)
            if current is None or row_score(row) > row_score(current):
                best[sample_id] = row
        accepted = list(best.values())

    accepted.sort(key=lambda row: str(row.get("id", "")))
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "sft_messages.jsonl", accepted)
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    report = {
        "input": args.sft,
        "output_dir": str(output_dir),
        "input_rows": len(rows),
        "accepted_before_best_of": accepted_before_best_of,
        "accepted_rows": len(accepted),
        "duplicate_rollouts_dropped": accepted_before_best_of - len(accepted),
        "rejected_rows": len(rejected),
        "best_of_sample": args.best_of_sample,
        "thresholds": {
            "min_overall": args.min_overall,
            "min_final_answer_chars": args.min_final_answer_chars,
            "min_citations": args.min_citations,
            "min_evidence_results": args.min_evidence_results,
            "max_tool_error_ratio": args.max_tool_error_ratio,
            "max_turns": args.max_turns,
            "allow_gate_failed": args.allow_gate_failed,
        },
        "rejection_reasons": dict(reason_counts),
        "category": dict(Counter((row.get("metadata", {}) or {}).get("task", {}).get("category", "") for row in accepted)),
        "difficulty": dict(Counter((row.get("metadata", {}) or {}).get("task", {}).get("difficulty", "") for row in accepted)),
    }
    (output_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
