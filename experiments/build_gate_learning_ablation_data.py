#!/usr/bin/env python3
"""Build process-SFT datasets for the gate-decision learning ablation.

The input is one or more RAES ``step_messages.jsonl`` files exported by
``src.raes_eval.training_export``.  The script writes two paired datasets:

* ``with_gate``: tool_call + gate_decision + final_answer samples
* ``no_gate``: tool_call + final_answer samples only

Both variants use the same parent trajectories and the same deterministic
train/validation split, so the only intended difference is whether
gate-decision supervision is present.
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


KEEP_WITH_GATE = {"tool_call", "gate_decision", "final_answer"}
KEEP_NO_GATE = {"tool_call", "final_answer"}
URL_RE = re.compile(r"https?://[^\s\)\]\}\>\"]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input step_messages.jsonl files.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument(
        "--exclude-question-files",
        nargs="*",
        default=[],
        help="JSONL files with held-out ids/questions to exclude from training rows.",
    )
    parser.add_argument(
        "--exclude-url-files",
        nargs="*",
        default=[],
        help="JSONL files whose URLs should be excluded from training rows.",
    )
    parser.add_argument(
        "--messages-only",
        action="store_true",
        help="Write only the messages field, matching common Swift SFT JSONL input.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def split_name(row: dict[str, Any], seed: int, val_percent: int) -> str:
    metadata = row.get("metadata") or {}
    parent_id = str(metadata.get("parent_id") or row.get("id") or "")
    digest = hashlib.sha256(f"{seed}:{parent_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "val" if bucket < val_percent else "train"


def sample_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("sample_type") or "")


def normalize_question(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def row_question(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if "Original RAES task:" in content:
            question = content.split("Original RAES task:", 1)[1]
            question = question.split("Trajectory context", 1)[0]
            return normalize_question(question)
        return normalize_question(content)
    return ""


def collect_question_exclusions(paths: list[str]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    questions: set[str] = set()
    for raw_path in paths:
        for row in load_jsonl(Path(raw_path)):
            row_id = str(row.get("id") or "")
            if row_id:
                ids.add(row_id)
            question = normalize_question(str(row.get("question") or ""))
            if question:
                questions.add(question)
    return ids, questions


def normalize_url(url: str) -> str:
    return url.split("#", 1)[0].rstrip("/").lower().rstrip(".,;")


def collect_urls(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
            collect_urls(value, out)
    elif isinstance(obj, list):
        for value in obj:
            collect_urls(value, out)
    elif isinstance(obj, str):
        for url in URL_RE.findall(obj):
            out.add(normalize_url(url))


def collect_url_exclusions(paths: list[str]) -> set[str]:
    urls: set[str] = set()
    for raw_path in paths:
        for row in load_jsonl(Path(raw_path)):
            collect_urls(row, urls)
    return urls


def row_urls(row: dict[str, Any]) -> set[str]:
    urls: set[str] = set()
    collect_urls(row, urls)
    return urls


def normalize(row: dict[str, Any], *, messages_only: bool) -> dict[str, Any]:
    if messages_only:
        return {"messages": row.get("messages") or []}
    return row


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids = 0
    excluded_ids, excluded_questions = collect_question_exclusions(args.exclude_question_files)
    excluded_urls = collect_url_exclusions(args.exclude_url_files)
    skipped_by_id = 0
    skipped_by_question = 0
    skipped_by_url = 0

    for raw_path in args.inputs:
        path = Path(raw_path)
        for row in load_jsonl(path):
            metadata = row.get("metadata") or {}
            sample_id = str(metadata.get("sample_id") or "")
            if sample_id and sample_id in excluded_ids:
                skipped_by_id += 1
                continue
            question = row_question(row)
            if question and question in excluded_questions:
                skipped_by_question += 1
                continue
            if excluded_urls and row_urls(row).intersection(excluded_urls):
                skipped_by_url += 1
                continue
            row_id = str(row.get("id") or "")
            if row_id and row_id in seen_ids:
                duplicate_ids += 1
                continue
            if row_id:
                seen_ids.add(row_id)
            rows.append(row)

    stats: dict[str, Any] = {
        "inputs": [str(Path(path)) for path in args.inputs],
        "input_rows_after_dedup": len(rows),
        "duplicate_ids_skipped": duplicate_ids,
        "heldout_question_files": [str(Path(path)) for path in args.exclude_question_files],
        "heldout_url_files": [str(Path(path)) for path in args.exclude_url_files],
        "heldout_ids": len(excluded_ids),
        "heldout_questions": len(excluded_questions),
        "heldout_urls": len(excluded_urls),
        "rows_skipped_by_heldout_id": skipped_by_id,
        "rows_skipped_by_heldout_question": skipped_by_question,
        "rows_skipped_by_heldout_url": skipped_by_url,
        "sample_type_counts": dict(Counter(sample_type(row) for row in rows)),
        "val_percent": args.val_percent,
        "seed": args.seed,
        "messages_only": bool(args.messages_only),
    }

    for variant, keep_types in {
        "with_gate": KEEP_WITH_GATE,
        "no_gate": KEEP_NO_GATE,
    }.items():
        variant_rows = [row for row in rows if sample_type(row) in keep_types]
        split_rows = {"train": [], "val": []}
        for row in variant_rows:
            split_rows[split_name(row, args.seed, args.val_percent)].append(
                normalize(row, messages_only=args.messages_only)
            )
        variant_dir = output_dir / variant
        counts = {
            "train": write_jsonl(variant_dir / "train.jsonl", split_rows["train"]),
            "val": write_jsonl(variant_dir / "val.jsonl", split_rows["val"]),
            "sample_type_counts": dict(Counter(sample_type(row) for row in variant_rows)),
        }
        stats[variant] = counts

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "build_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_dir / "build_report.json")


if __name__ == "__main__":
    main()
