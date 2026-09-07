#!/usr/bin/env python3
"""Run a RAES-Bench split with a configured agent."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.raes_eval.dataset import RAESDataset
from src.raes_eval.runner import AgentConfig, RAESRunner, load_config
from src.core.errors import ApiQuotaExceeded, ExternalServiceFailure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="raes-bench/raes-bench-v12_process/dev_with_gold.jsonl",
        help="RAES JSONL path. Use public/test files for gold-hidden runs.",
    )
    parser.add_argument("--config", default="experiments/configs/dry_run.yaml")
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress output.")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = RAESDataset(
        args.dataset,
        split=args.split or None,
        limit=args.limit or None,
        sample=args.sample or None,
        seed=args.seed,
    )
    dataset.assert_no_gold_leakage()
    runner = RAESRunner(
        dataset=dataset,
        agent_config=AgentConfig.from_dict(config),
        output_dir=args.output_dir,
        run_id=args.run_id or None,
        resume=not args.no_resume,
        show_progress=not args.no_progress,
    )
    await runner.run()
    print(runner.results_path)


def main() -> None:
    try:
        asyncio.run(main_async())
    except ApiQuotaExceeded as exc:
        print(f"ABORTED: API quota/rate/billing limit hit: {exc}", file=sys.stderr)
        print(
            "Completed samples already written to results.jsonl are preserved. "
            "Re-run the same command with the same --run-id and without --no-resume to continue.",
            file=sys.stderr,
        )
        raise SystemExit(75)
    except ExternalServiceFailure as exc:
        print(f"ABORTED: external service failure: {exc}", file=sys.stderr)
        print(
            "Completed samples already written to results.jsonl are preserved. "
            "The current sample was not written; re-run the same command with "
            "the same --run-id and without --no-resume to retry it.",
            file=sys.stderr,
        )
        raise SystemExit(75)


if __name__ == "__main__":
    main()
