#!/usr/bin/env python3
"""Refuse to resume a RAES run whose stored config differs from the current one."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.raes_eval.runner import AgentConfig, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = AgentConfig.from_dict(load_config(args.config)).to_dict()
    trajectory_dir = Path(args.run_dir) / "trajectories"
    checked = 0
    mismatches = []
    for path in sorted(trajectory_dir.glob("*.trajectory.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            first_line = next((line for line in handle if line.strip()), "")
        if not first_line:
            continue
        first = json.loads(first_line)
        stored = ((first.get("data") or {}).get("agent_config"))
        if first.get("event_type") != "sample_start" or not isinstance(stored, dict):
            mismatches.append({"trajectory": str(path), "reason": "missing sample_start agent_config"})
        elif stored != expected:
            mismatches.append({"trajectory": str(path), "reason": "agent_config differs"})
        checked += 1
    report = {
        "status": "passed" if not mismatches else "failed",
        "config": str(Path(args.config)),
        "run_dir": str(Path(args.run_dir)),
        "existing_trajectories_checked": checked,
        "mismatches": mismatches,
    }
    print(json.dumps(report, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
