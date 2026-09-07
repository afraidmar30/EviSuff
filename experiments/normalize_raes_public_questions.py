#!/usr/bin/env python3
"""Rewrite RAES public questions that contain benchmark/meta search wording.

The rewrite is deterministic and gold-free: it only uses public fields already
present in each row.  It preserves the task target and boundary focus while
removing phrases that are useful as benchmark labels but poor web-search text.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_PATHS = [
    "raes-bench/raes-bench-v12_process/public_questions.jsonl",
    "raes-bench/raes-bench-v12_process/public_questions_1k.jsonl",
    "raes-bench/raes-bench-v12_process/public_questions_1k_synthetic_additions.jsonl",
]

META_PATTERNS = (
    "reliable public evidence boundary",
    "public evidence boundary",
    "evidence boundary",
    "benchmark evidence boundary",
    "single-source verification task",
    "medium data_analysis task",
    "easy deep_search",
    "hard deep_search task",
    "technical documentation lookup task",
)

FOCUS_LABELS = {
    "primary-source identity": "primary-source identity",
    "claim-scope boundary": "claim scope",
    "implementation locator": "implementation locator",
    "evaluation-protocol locator": "evaluation protocol locator",
    "temporal-validity boundary": "temporal validity",
    "evidence-sufficiency boundary": "evidence sufficiency",
    "conflict-resolution boundary": "conflict resolution",
    "citation-faithfulness boundary": "citation faithfulness",
    "source-authority hierarchy": "source authority hierarchy",
    "public-access boundary": "public access",
    "reproducibility-artifact boundary": "reproducibility artifacts",
    "limitation-caveat boundary": "limitations and caveats",
    "negative-evidence boundary": "negative evidence",
}

TEMPLATE_RE = re.compile(
    r"^As of (?P<date>\d{4}-\d{2}-\d{2}),\s*"
    r"verify the (?P<label>.*?):\s*"
    r"what is the reliable public evidence boundary for (?P<evidence>.*?) "
    r"for (?P<target>.*?) with a (?P<focus>.*?) focus\?$",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="*", default=DEFAULT_PATHS)
    parser.add_argument("--shard-dir", default="raes-bench/raes-bench-v12_process/shards_1k")
    parser.add_argument("--shard-glob", default="public_questions_1k_shard*.jsonl")
    parser.add_argument("--report", default="raes-bench/raes-bench-v12_process/question_normalization_report.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def has_meta(question: str) -> bool:
    lower = question.lower()
    return any(pattern in lower for pattern in META_PATTERNS)


def clean_focus(raw: str) -> str:
    raw = raw.strip().replace("_", " ")
    return FOCUS_LABELS.get(raw.lower(), raw.replace("-", " "))


def task_goal(row: dict[str, Any], evidence: str, focus: str) -> str:
    category = str(row.get("category", ""))
    task_type = str(row.get("task_type", ""))
    focus = clean_focus(focus)
    if category == "technical":
        return (
            "Verify implementation and version evidence from official documentation, "
            f"GitHub repositories, releases, changelogs, package pages, or papers. Focus on {focus}."
        )
    if category == "data_analysis":
        return (
            "Verify documentation and version evidence relevant to data-analysis API, metric, "
            f"or reproducibility claims. Focus on {focus}."
        )
    if task_type == "leaderboard_verification":
        return (
            "Verify the correct public source for leaderboard-style claims and distinguish official "
            f"sources from mirrors or aggregators. Focus on {focus}."
        )
    if task_type == "multi_source_timeline_reconstruction":
        return (
            "Reconstruct the public evidence timeline from direct sources such as papers, project pages, "
            f"GitHub, dataset pages, or release notes. Focus on {focus}."
        )
    if task_type == "adversarial_conflict_resolution":
        return (
            "Compare public sources, identify conflicts or limitations, and state what can be concluded. "
            f"Focus on {focus}."
        )
    if "deep-search" in evidence.lower() or category == "deep_search":
        return (
            "Verify the public source boundary for the benchmark, dataset, paper, project page, "
            f"implementation, or evaluation protocol. Focus on {focus}."
        )
    return f"Verify the public-source support for the target. Focus on {focus}."


def rewrite_template(row: dict[str, Any], match: re.Match[str]) -> str:
    date = match.group("date")
    target = re.sub(r"\s+", " ", match.group("target")).strip()
    evidence = match.group("evidence")
    focus = match.group("focus")
    goal = task_goal(row, evidence, focus)
    return (
        f"As of {date}, investigate {target}. {goal} "
        "Identify the primary public sources, what claims they support, what version/date or scope "
        "boundaries apply, and what remains uncertain."
    )


def fallback_rewrite(row: dict[str, Any], question: str) -> str:
    q = question
    replacements = [
        (r"\bnegative-evidence boundary\b", "negative evidence"),
        (r"\bevidence-sufficiency boundary\b", "evidence sufficiency"),
        (r"\btemporal-validity boundary\b", "temporal validity"),
        (r"\breproducibility-artifact boundary\b", "reproducibility artifacts"),
        (r"\blimitation-caveat boundary\b", "limitations and caveats"),
        (r"\bconflict-resolution boundary\b", "conflict resolution"),
        (r"\bcitation-faithfulness boundary\b", "citation faithfulness"),
        (r"\bpublic-access boundary\b", "public access"),
        (r"\bclaim-scope boundary\b", "claim scope"),
        (r"verify the (easy|medium|hard) [^:]* task:\s*", "verify "),
        (r"what is the reliable public evidence boundary for ", "the public-source support for "),
        (r"deep-search benchmark evidence boundary for ", ""),
        (r"data-analysis evidence boundary for ", ""),
        (r"implementation or version evidence for ", "implementation and version evidence for "),
        (r"\bwith a ([^?]*?) focus\?", r"with focus on \1."),
        (r"\breliable public evidence boundary\b", "public-source support"),
        (r"\bevidence boundary\b", "source support"),
    ]
    for pattern, repl in replacements:
        q = re.sub(pattern, repl, q, flags=re.I)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def normalize_row(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    question = str(row.get("question", ""))
    if not has_meta(question):
        return row, False
    match = TEMPLATE_RE.match(question)
    rewritten = rewrite_template(row, match) if match else fallback_rewrite(row, question)
    rewritten = re.sub(r"\s+", " ", rewritten).strip()
    if rewritten == question:
        return row, False
    updated = dict(row)
    updated.setdefault("original_question", question)
    updated["question"] = rewritten
    updated["question_normalized"] = True
    updated["question_normalization_version"] = "search_friendly_v1"
    return updated, True


def process_file(path: Path, *, dry_run: bool) -> dict[str, Any]:
    rows = load_jsonl(path)
    changed = 0
    examples: list[dict[str, str]] = []
    out: list[dict[str, Any]] = []
    for row in rows:
        new_row, did_change = normalize_row(row)
        out.append(new_row)
        if did_change:
            changed += 1
            if len(examples) < 8:
                examples.append({
                    "id": str(row.get("id", "")),
                    "before": str(row.get("question", "")),
                    "after": str(new_row.get("question", "")),
                })
    if changed and not dry_run:
        write_jsonl(path, out)
    return {"path": str(path), "rows": len(rows), "changed": changed, "examples": examples}


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.paths]
    shard_dir = Path(args.shard_dir)
    if shard_dir.exists():
        paths.extend(sorted(shard_dir.glob(args.shard_glob)))
    # Preserve order while removing duplicates.
    unique_paths = list(dict.fromkeys(paths))
    reports = [process_file(path, dry_run=args.dry_run) for path in unique_paths if path.exists()]
    summary = {
        "dry_run": args.dry_run,
        "files": len(reports),
        "total_rows": sum(item["rows"] for item in reports),
        "total_changed": sum(item["changed"] for item in reports),
        "by_file": reports,
        "changed_file_count": Counter(bool(item["changed"]) for item in reports)[True],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.dry_run:
        Path(args.report).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
