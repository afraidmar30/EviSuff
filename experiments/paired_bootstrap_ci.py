#!/usr/bin/env python3
"""Compute item-level bootstrap confidence intervals for RAES score files.

The script reports per-run mean CIs and paired delta CIs over shared sample ids.
It is intended for paper tables where systems are evaluated on the same item set.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = ("answer_correctness", "facet_coverage", "evidence_support", "overall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", nargs="+", required=True, help="Score JSONL files.")
    parser.add_argument("--labels", nargs="*", default=[], help="Optional labels, one per score file.")
    parser.add_argument("--metrics", nargs="*", default=list(METRICS))
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or row.get("id") or "")
            if not sample_id:
                continue
            out[sample_id] = {
                metric: float(row.get(metric, 0.0) or 0.0)
                for metric in METRICS
            }
    return out


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def ci(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(mean(values), 4) if values else 0.0,
        "ci_low": round(percentile(values, 0.025), 4),
        "ci_high": round(percentile(values, 0.975), 4),
    }


def bootstrap_mean(
    rows: dict[str, dict[str, float]],
    metric: str,
    iterations: int,
    rng: random.Random,
) -> dict[str, float]:
    ids = list(rows)
    observed = mean(rows[item][metric] for item in ids) if ids else 0.0
    draws = []
    for _ in range(iterations):
        sample = [rng.choice(ids) for _ in ids]
        draws.append(mean(rows[item][metric] for item in sample))
    result = ci(draws)
    result["observed"] = round(observed, 4)
    result["n"] = len(ids)
    return result


def bootstrap_delta(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    metric: str,
    iterations: int,
    rng: random.Random,
) -> dict[str, float]:
    ids = sorted(set(left) & set(right))
    observed = mean(left[item][metric] - right[item][metric] for item in ids) if ids else 0.0
    draws = []
    for _ in range(iterations):
        sample = [rng.choice(ids) for _ in ids]
        draws.append(mean(left[item][metric] - right[item][metric] for item in sample))
    result = ci(draws)
    result["observed"] = round(observed, 4)
    result["n"] = len(ids)
    return result


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# Paired Bootstrap Confidence Intervals", ""]
    lines.append("## Per-Run Means")
    lines.append("")
    lines.append("| Run | Metric | N | Mean | 95% CI |")
    lines.append("|---|---:|---:|---:|---:|")
    for run, metrics in report["runs"].items():
        for metric, values in metrics.items():
            lines.append(
                f"| {run} | {metric} | {values['n']} | {values['observed']:.4f} | "
                f"[{values['ci_low']:.4f}, {values['ci_high']:.4f}] |"
            )
    lines.append("")
    lines.append("## Paired Deltas")
    lines.append("")
    lines.append("| Contrast | Metric | N | Delta | 95% CI |")
    lines.append("|---|---:|---:|---:|---:|")
    for contrast, metrics in report["paired_deltas"].items():
        for metric, values in metrics.items():
            lines.append(
                f"| {contrast} | {metric} | {values['n']} | {values['observed']:.4f} | "
                f"[{values['ci_low']:.4f}, {values['ci_high']:.4f}] |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    labels = args.labels or [Path(path).stem for path in args.scores]
    if len(labels) != len(args.scores):
        raise SystemExit("--labels must have the same length as --scores")

    rng = random.Random(args.seed)
    runs = {
        label: load_scores(Path(path))
        for label, path in zip(labels, args.scores)
    }
    report: dict[str, Any] = {
        "iterations": args.iterations,
        "seed": args.seed,
        "runs": {},
        "paired_deltas": {},
    }
    for label, rows in runs.items():
        report["runs"][label] = {
            metric: bootstrap_mean(rows, metric, args.iterations, rng)
            for metric in args.metrics
        }
    label_list = list(runs)
    for i, left_label in enumerate(label_list):
        for right_label in label_list[i + 1:]:
            contrast = f"{left_label} - {right_label}"
            report["paired_deltas"][contrast] = {
                metric: bootstrap_delta(
                    runs[left_label],
                    runs[right_label],
                    metric,
                    args.iterations,
                    rng,
                )
                for metric in args.metrics
            }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.output_md))
    print(output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()

