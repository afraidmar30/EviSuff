#!/usr/bin/env python3
"""Build a held-out, answerability-balanced task set for sufficiency audits."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GROUP_MAP = {
    "answerable": "answerable",
    "answerable_with_uncertainty": "uncertainty",
    "conflicting_no_resolution": "conflict",
    "not_enough_public_evidence": "insufficient-evidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output-questions", required=True)
    parser.add_argument("--output-gold", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--splits", nargs="+", default=["test", "stress"])
    parser.add_argument("--answerable", type=int, default=7)
    parser.add_argument("--uncertainty", type=int, default=6)
    parser.add_argument("--conflict", type=int, default=6)
    parser.add_argument("--insufficient-evidence", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--exclude-message-jsonl",
        nargs="*",
        default=[],
        help=(
            "Training/validation JSONL files containing chat `messages`. "
            "A candidate is excluded when its normalized question appears "
            "verbatim inside any user message."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def excluded_question_ids(
    questions: list[dict[str, Any]], paths: list[str]
) -> tuple[set[str], dict[str, Any]]:
    normalized_questions = {
        str(row.get("id") or ""): norm_text(row.get("question"))
        for row in questions
        if row.get("id") and row.get("question")
    }
    excluded: set[str] = set()
    per_file: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        matches: set[str] = set()
        rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
                user_text = norm_text("\n".join(
                    str(message.get("content") or "")
                    for message in row.get("messages") or []
                    if message.get("role") == "user"
                ))
                for sample_id, question in normalized_questions.items():
                    if sample_id not in excluded and question and question in user_text:
                        matches.add(sample_id)
        excluded.update(matches)
        per_file[str(path)] = {
            "rows_scanned": rows,
            "new_question_ids_excluded": sorted(matches),
            "new_question_count": len(matches),
        }
    return excluded, per_file


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    questions = load_jsonl(Path(args.questions))
    gold_rows = load_jsonl(Path(args.gold))
    gold = {str(row["id"]): row for row in gold_rows}
    excluded_ids, exclusion_report = excluded_question_ids(
        questions, args.exclude_message_jsonl
    )
    quotas = {
        "answerable": args.answerable,
        "uncertainty": args.uncertainty,
        "conflict": args.conflict,
        "insufficient-evidence": args.insufficient_evidence,
    }
    if any(value < 0 for value in quotas.values()) or not sum(quotas.values()):
        raise SystemExit("Quotas must be non-negative and sum to at least one")

    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_set = set(args.splits)
    for question in questions:
        sid = str(question.get("id") or "")
        if (
            sid not in gold
            or sid in excluded_ids
            or str(question.get("split") or "") not in split_set
        ):
            continue
        group = GROUP_MAP.get(str(gold[sid].get("answerability") or ""))
        if group:
            eligible[group].append(question)

    selected: list[dict[str, Any]] = []
    for group, quota in quotas.items():
        candidates = eligible[group]
        if len(candidates) < quota:
            raise SystemExit(f"{group}: requested {quota}, only {len(candidates)} eligible")
        rng.shuffle(candidates)
        selected.extend(candidates[:quota])
    rng.shuffle(selected)
    selected_gold = [gold[str(row["id"])] for row in selected]

    write_jsonl(Path(args.output_questions), selected)
    write_jsonl(Path(args.output_gold), selected_gold)
    summary = {
        "seed": args.seed,
        "source_questions": args.questions,
        "source_gold": args.gold,
        "eligible_splits": args.splits,
        "held_out_only": set(args.splits) <= {"test", "stress"},
        "training_message_files_scanned": args.exclude_message_jsonl,
        "training_overlap_excluded_ids": sorted(excluded_ids),
        "training_overlap_excluded_count": len(excluded_ids),
        "training_overlap_scan": exclusion_report,
        "task_count": len(selected),
        "quotas": quotas,
        "selected_ids": [row["id"] for row in selected],
        "selected_split_counts": dict(Counter(str(row.get("split") or "") for row in selected)),
        "selected_group_counts": dict(Counter(
            GROUP_MAP[str(gold[str(row["id"])]["answerability"])] for row in selected
        )),
    }
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
