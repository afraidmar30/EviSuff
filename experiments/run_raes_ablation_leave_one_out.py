#!/usr/bin/env python3
"""Run the RAES-Gate leave-one-out ablation suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        default="experiments/configs/raes/leave_one_out_ablation_suite.yaml",
        help="YAML suite with runs[].",
    )
    parser.add_argument(
        "--dataset",
        default="raes-bench/raes-bench-v12_process/public_questions_stratified20_seed20260520.jsonl",
    )
    parser.add_argument("--gold", default="raes-bench/raes-bench-v12_process/gold.jsonl")
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--scores-dir", default="outputs/scores")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--prefix", default="leave_one_out")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--resume", action="store_true", help="Resume existing run IDs instead of starting fresh.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    return parser.parse_args()


def load_suite(path: str | Path) -> list[dict]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for YAML experiment configs") from exc

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    runs = data.get("runs") or []
    if not runs:
        raise ValueError(f"suite has no runs: {path}")
    return runs


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    runs = load_suite(args.suite)
    score_paths: list[str] = []

    for run in runs:
        name = str(run["name"])
        config = str(run["config"])
        run_id = f"{args.prefix}_{name}"

        run_cmd_args = [
            sys.executable,
            "experiments/run_raes.py",
            "--config",
            config,
            "--dataset",
            args.dataset,
            "--output-dir",
            args.output_dir,
            "--run-id",
            run_id,
            "--seed",
            str(args.seed),
        ]
        if args.limit:
            run_cmd_args.extend(["--limit", str(args.limit)])
        if args.sample:
            run_cmd_args.extend(["--sample", str(args.sample)])
        if not args.resume:
            run_cmd_args.append("--no-resume")
        if args.no_progress:
            run_cmd_args.append("--no-progress")
        run_cmd(run_cmd_args)

        if args.skip_score:
            continue
        score_path = str(Path(args.scores_dir) / f"{run_id}.scores.jsonl")
        score_paths.append(score_path)
        run_cmd([
            sys.executable,
            "experiments/score_raes.py",
            "--gold",
            args.gold,
            "--results",
            str(Path(args.output_dir) / run_id / "results.jsonl"),
            "--output",
            score_path,
        ])

    if score_paths and not args.skip_score:
        run_cmd([
            sys.executable,
            "experiments/aggregate_results.py",
            "--scores",
            *score_paths,
            "--output-json",
            str(Path(args.tables_dir) / f"{args.prefix}.summary.json"),
            "--output-md",
            str(Path(args.tables_dir) / f"{args.prefix}.summary.md"),
        ])


if __name__ == "__main__":
    main()
