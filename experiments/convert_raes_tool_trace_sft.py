#!/usr/bin/env python3
"""Convert filtered RAES tool traces to native ms-swift tool-call SFT rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.academic_search import AcademicSearchTool
from src.tools.cite_source import CiteSourceTool
from src.tools.deep_read import DeepReadTool
from src.tools.news_search import NewsSearchTool
from src.tools.web_fetch import WebFetchTool
from src.tools.web_search import WebSearchTool
from src.tools.wechat_search import WeChatSearchTool


TOOL_CLASSES = (
    WebSearchTool,
    WebFetchTool,
    DeepReadTool,
    CiteSourceTool,
    AcademicSearchTool,
    NewsSearchTool,
    WeChatSearchTool,
)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": cls.name,
            "description": cls.description,
            "parameters": cls.input_schema,
        },
    }
    for cls in TOOL_CLASSES
]
ALLOWED_TOOLS = {schema["function"]["name"] for schema in TOOL_SCHEMAS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Strict-pass sft_messages.jsonl with tool_trace.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tool-result-chars", type=int, default=8000)
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=20260519)
    parser.add_argument("--tokenizer", help="Optional tokenizer.json used for approximate length statistics.")
    parser.add_argument("--max-estimated-tokens", type=int, default=40000)
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def swift_training_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Drop audit metadata whose sparse dictionaries confuse Arrow schema inference."""
    for row in rows:
        yield {"messages": row["messages"], "tools": row["tools"]}


def clean_system_prompt(text: str) -> str:
    """Remove prose tool catalogs and instructions for tools disabled in the target config."""
    text = re.sub(
        r"\n## Available Tools\n.*?(?=\n## Citation Guidelines\n)",
        "\n",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\n## IMPORTANT: Structured Research Plans\n.*?(?=\n## Current Date\n)",
        "\n",
        text,
        flags=re.DOTALL,
    )
    lines = []
    for line in text.splitlines():
        lowered = line.lower()
        if "research_plan" in lowered:
            continue
        if "update the research plan" in lowered:
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def split_is_validation(sample_id: str, seed: int, val_percent: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16)
    # Preserve the exact 90/10 split used by the answer-SFT dataset.
    if val_percent == 10:
        return bucket % 10 == 0
    return bucket % 100 < val_percent


def convert_row(row: dict[str, Any], max_tool_result_chars: int) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    source_messages = row.get("messages") or []
    system = next((str(m.get("content") or "") for m in source_messages if m.get("role") == "system"), "")
    user = next((str(m.get("content") or "") for m in source_messages if m.get("role") == "user"), "")
    final_answer = str(row.get("final_answer") or "").strip()
    trace = row.get("tool_trace") or []
    if not user:
        reasons.append("missing_user_message")
    if not final_answer:
        reasons.append("missing_final_answer")

    results_by_id = {
        str(item.get("tool_use_id")): item
        for item in trace
        if item.get("type") == "tool_result" and item.get("tool_use_id")
    }
    retained_call_ids: set[str] = set()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": clean_system_prompt(system)})
    messages.append({"role": "user", "content": user})

    dropped_tools: Counter[str] = Counter()
    unmatched_tools: Counter[str] = Counter()
    for item in trace:
        item_type = item.get("type")
        if item_type == "tool_call":
            tool_name = str(item.get("tool_name") or "")
            call_id = str(item.get("tool_use_id") or "")
            if tool_name not in ALLOWED_TOOLS:
                dropped_tools[tool_name or "<missing>"] += 1
                continue
            if not call_id or call_id not in results_by_id:
                unmatched_tools[tool_name] += 1
                continue
            tool_input = item.get("tool_input")
            if not isinstance(tool_input, dict):
                unmatched_tools[tool_name] += 1
                continue
            content = json.dumps(
                {"name": tool_name, "arguments": tool_input},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            messages.append({"role": "tool_call", "content": content})
            retained_call_ids.add(call_id)
        elif item_type == "tool_result":
            call_id = str(item.get("tool_use_id") or "")
            if call_id not in retained_call_ids:
                continue
            result = str(item.get("result") or "")
            if max_tool_result_chars > 0 and len(result) > max_tool_result_chars:
                result = result[:max_tool_result_chars].rstrip() + "\n\n[Tool result truncated for training.]"
            messages.append({"role": "tool", "content": result})

    messages.append({"role": "assistant", "content": final_answer})
    tool_calls = sum(message["role"] == "tool_call" for message in messages)
    tool_results = sum(message["role"] == "tool" for message in messages)
    if tool_calls == 0:
        reasons.append("no_supported_complete_tool_calls")
    if tool_calls != tool_results:
        reasons.append("tool_call_result_count_mismatch")
    if reasons:
        return None, reasons

    metadata = row.get("metadata") or {}
    converted = {
        "id": row.get("id"),
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "metadata": {
            "run_id": metadata.get("run_id"),
            "sample_id": metadata.get("sample_id"),
            "source_id": row.get("id"),
            "tool_calls": tool_calls,
            "dropped_tools": dict(dropped_tools),
            "unmatched_tools": dict(unmatched_tools),
        },
    }
    return converted, []


def approximate_render(row: dict[str, Any]) -> str:
    """Render enough of Hermes/ChatML to provide a conservative tokenizer estimate."""
    parts: list[str] = []
    tools = row.get("tools") or []
    for message in row.get("messages") or []:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "system":
            parts.append("<|im_start|>system\n" + content)
            parts.append("\n# Tools\n" + json.dumps(tools, ensure_ascii=False))
            parts.append("<|im_end|>\n")
        elif role == "tool_call":
            parts.append("<|im_start|>assistant\n<tool_call>\n" + content + "\n</tool_call><|im_end|>\n")
        elif role == "tool":
            parts.append("<|im_start|>user\n<tool_response>\n" + content + "\n</tool_response><|im_end|>\n")
        else:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def percentile(values: list[int], p: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))]


def main() -> None:
    args = parse_args()
    if not 0 < args.val_percent < 100:
        raise SystemExit("--val-percent must be between 1 and 99")
    tokenizer = None
    if args.tokenizer:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(args.tokenizer)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    dropped_tool_counts: Counter[str] = Counter()
    unmatched_tool_counts: Counter[str] = Counter()
    token_lengths: list[int] = []
    total = 0
    for source_row in load_jsonl(Path(args.input)):
        total += 1
        converted, reasons = convert_row(source_row, args.max_tool_result_chars)
        if converted is not None and tokenizer is not None:
            estimated_tokens = len(tokenizer.encode(approximate_render(converted)).ids)
            converted["metadata"]["estimated_tokens"] = estimated_tokens
            if estimated_tokens > args.max_estimated_tokens:
                reasons = ["estimated_context_too_long"]
                converted = None
            else:
                token_lengths.append(estimated_tokens)
        if converted is None:
            reason_counts.update(reasons)
            rejected.append({"id": source_row.get("id"), "reasons": reasons})
            continue
        dropped_tool_counts.update(converted["metadata"].get("dropped_tools") or {})
        unmatched_tool_counts.update(converted["metadata"].get("unmatched_tools") or {})
        accepted.append(converted)

    accepted.sort(key=lambda row: str(row.get("id") or ""))
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    train_sample_ids: set[str] = set()
    val_sample_ids: set[str] = set()
    for row in accepted:
        sample_id = str((row.get("metadata") or {}).get("sample_id") or row.get("id") or "")
        if split_is_validation(sample_id, args.split_seed, args.val_percent):
            val.append(row)
            val_sample_ids.add(sample_id)
        else:
            train.append(row)
            train_sample_ids.add(sample_id)
    if train_sample_ids & val_sample_ids:
        raise RuntimeError("sample_id leakage detected between train and validation splits")

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "tool_sft_messages.jsonl", accepted)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)
    write_jsonl(output_dir / "train_swift.jsonl", swift_training_rows(train))
    write_jsonl(output_dir / "val_swift.jsonl", swift_training_rows(val))
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    report: dict[str, Any] = {
        "input": str(Path(args.input)),
        "output_dir": str(output_dir),
        "input_rows": total,
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "train_rows": len(train),
        "val_rows": len(val),
        "train_sample_ids": len(train_sample_ids),
        "val_sample_ids": len(val_sample_ids),
        "split_leakage_sample_ids": 0,
        "max_tool_result_chars": args.max_tool_result_chars,
        "allowed_tools": sorted(ALLOWED_TOOLS),
        "rejection_reasons": dict(reason_counts),
        "dropped_tool_calls": dict(dropped_tool_counts),
        "unmatched_tool_calls": dict(unmatched_tool_counts),
    }
    if token_lengths:
        report["estimated_token_stats"] = {
            "p50": percentile(token_lengths, 0.50),
            "p90": percentile(token_lengths, 0.90),
            "p95": percentile(token_lengths, 0.95),
            "p99": percentile(token_lengths, 0.99),
            "max": max(token_lengths),
            "limit": args.max_estimated_tokens,
        }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
