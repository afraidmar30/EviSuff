#!/usr/bin/env python3
"""Export RAES run trajectories to SFT-ready JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.raes_eval.training_export import export_sft_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="",
        help="Run directory containing results.jsonl and trajectories/.",
    )
    parser.add_argument("--results", default="", help="Explicit results.jsonl path.")
    parser.add_argument("--trajectories", default="", help="Explicit trajectory directory.")
    parser.add_argument("--output-dir", default="", help="Output directory for training artifacts.")
    parser.add_argument("--run-id", default="", help="Run id to write into reports.")
    parser.add_argument(
        "--allow-no-evidence",
        action="store_true",
        help="Do not reject samples without citation/source evidence.",
    )
    parser.add_argument(
        "--keep-forced-gate",
        action="store_true",
        help="Keep samples that RAES-Gate forced through after budget exhaustion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is None and not args.results:
        raise SystemExit("Provide --run-dir or --results.")

    results = Path(args.results) if args.results else run_dir / "results.jsonl"
    trajectories = (
        Path(args.trajectories)
        if args.trajectories
        else (run_dir / "trajectories" if run_dir else results.parent / "trajectories")
    )
    run_id = args.run_id or (run_dir.name if run_dir else results.parent.name)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/training") / run_id

    outputs = export_sft_dataset(
        results_path=results,
        trajectory_dir=trajectories,
        output_dir=output_dir,
        run_id=run_id,
        allow_no_evidence=args.allow_no_evidence,
        keep_forced_gate=args.keep_forced_gate,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
