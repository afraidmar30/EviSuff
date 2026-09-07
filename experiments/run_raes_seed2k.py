#!/usr/bin/env python3
"""Run multi-rollout RAES trajectory collection and merge SFT exports."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.raes_eval.runner import load_config
from src.raes_eval.training_export import export_sft_dataset, merge_sft_exports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="raes-bench/raes-bench-v12_process/public_questions.jsonl")
    parser.add_argument("--config", default="experiments/configs/searchclaw_raes_gate.yaml")
    parser.add_argument("--prefix", default="qwen_plus_searchclaw_raes_gate_seed2k")
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--temperatures", default="0,0.2,0.4,0.6")
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--training-dir", default="outputs/training")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Only export/merge existing rollout run dirs.")
    parser.add_argument("--skip-export", action="store_true", help="Run rollouts but skip export and merge.")
    parser.add_argument("--keep-forced-gate", action="store_true")
    parser.add_argument("--allow-no-evidence", action="store_true")
    return parser.parse_args()


def parse_temperatures(raw: str, rollouts: int) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        values = [0.0]
    while len(values) < rollouts:
        values.append(values[-1])
    return values[:rollouts]


def temp_label(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def write_rollout_config(base_config: dict, path: Path, *, temperature: float) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write rollout configs") from exc

    cfg = copy.deepcopy(base_config)
    cfg["temperature"] = temperature
    cfg.setdefault("trajectory", {})
    cfg["trajectory"].setdefault("capture_full_text", True)
    cfg["trajectory"].setdefault("max_tool_result_chars", 20000)
    cfg.setdefault("training_export", {})
    cfg["training_export"].setdefault("format", "messages_jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    temperatures = parse_temperatures(args.temperatures, args.rollouts)
    output_dir = Path(args.output_dir)
    training_dir = Path(args.training_dir)
    generated_config_dir = training_dir / args.prefix / "configs"
    export_dirs: list[Path] = []

    for rollout_idx, temperature in enumerate(temperatures):
        run_id = f"{args.prefix}_rollout{rollout_idx:02d}_t{temp_label(temperature)}"
        config_path = generated_config_dir / f"{run_id}.yaml"
        write_rollout_config(base_config, config_path, temperature=temperature)

        if not args.skip_run:
            cmd = [
                sys.executable,
                "experiments/run_raes.py",
                "--config",
                str(config_path),
                "--dataset",
                args.dataset,
                "--output-dir",
                str(output_dir),
                "--run-id",
                run_id,
                "--seed",
                str(args.seed + rollout_idx),
            ]
            if args.limit:
                cmd.extend(["--limit", str(args.limit)])
            if args.sample:
                cmd.extend(["--sample", str(args.sample)])
            if args.no_resume:
                cmd.append("--no-resume")
            if args.no_progress:
                cmd.append("--no-progress")
            run_cmd(cmd)

        if args.skip_export:
            continue
        run_dir = output_dir / run_id
        export_dir = training_dir / run_id
        export_sft_dataset(
            results_path=run_dir / "results.jsonl",
            trajectory_dir=run_dir / "trajectories",
            output_dir=export_dir,
            run_id=run_id,
            allow_no_evidence=args.allow_no_evidence,
            keep_forced_gate=args.keep_forced_gate,
        )
        export_dirs.append(export_dir)

    if export_dirs and not args.skip_export:
        outputs = merge_sft_exports(
            export_dirs=export_dirs,
            output_dir=training_dir / args.prefix,
            name=args.prefix,
        )
        for key, path in outputs.items():
            print(f"{key}: {path}")


if __name__ == "__main__":
    main()
