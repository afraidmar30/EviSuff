#!/usr/bin/env python3
"""Verify that no-gate data are an exact subset of with-gate data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--build-report", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def line_hashes(path: Path) -> Counter[str]:
    hashes: Counter[str] = Counter()
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                hashes[hashlib.sha256(line.rstrip(b"\r\n")).hexdigest()] += 1
    return hashes


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    build = json.loads(Path(args.build_report).read_text(encoding="utf-8"))
    expected_gate_rows = int((build.get("sample_type_counts") or {}).get("gate_decision", -1))
    checks = []
    extra_total = 0
    for split in ("train", "val"):
        with_gate = line_hashes(data_dir / "with_gate" / f"{split}.jsonl")
        no_gate = line_hashes(data_dir / "no_gate" / f"{split}.jsonl")
        missing = no_gate - with_gate
        extra = with_gate - no_gate
        extra_total += sum(extra.values())
        checks.append({
            "split": split,
            "with_gate_rows": sum(with_gate.values()),
            "no_gate_rows": sum(no_gate.values()),
            "no_gate_rows_missing_from_with_gate": sum(missing.values()),
            "extra_with_gate_rows": sum(extra.values()),
            "exact_subset": not missing,
        })
    passed = all(check["exact_subset"] for check in checks) and extra_total == expected_gate_rows
    report = {
        "status": "passed" if passed else "failed",
        "comparison": "no_gate is an exact multiset subset of with_gate",
        "line_identity": "SHA-256 of each complete JSONL row",
        "splits": checks,
        "extra_with_gate_rows_total": extra_total,
        "expected_gate_decision_rows": expected_gate_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
