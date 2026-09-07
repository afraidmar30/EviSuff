#!/usr/bin/env python3
"""Normalize external QA datasets into a RAES-like JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--output", required=True, help="Output RAES-like JSONL file.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--category-field", default="")
    parser.add_argument("--task-type", default="external_search_qa")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def get_path(row: dict[str, Any], dotted: str, default: Any = "") -> Any:
    if not dotted:
        return default
    value: Any = row
    for part in dotted.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, 1):
            raw_id = get_path(row, args.id_field, idx)
            question = str(get_path(row, args.question_field, "") or "").strip()
            answer = get_path(row, args.answer_field, "")
            if isinstance(answer, (list, tuple)):
                answer = "; ".join(str(item) for item in answer)
            answer = str(answer or "").strip()
            if not question:
                continue
            out = {
                "id": f"{args.dataset_name}-{raw_id}",
                "question": question,
                "category": str(get_path(row, args.category_field, args.dataset_name) or args.dataset_name),
                "task_type": args.task_type,
                "difficulty": "",
                "answerability": "answerable" if answer else "",
                "gold_answer": answer,
                "source_dataset": args.dataset_name,
                "source_record": row,
            }
            f.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(output)


if __name__ == "__main__":
    main()
