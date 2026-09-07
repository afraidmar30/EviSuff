#!/usr/bin/env python3
"""Run a YAML-defined RAES experiment suite."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="Experiment suite YAML.")
    parser.add_argument("--groups", nargs="*", default=[], help="Optional group names to run.")
    parser.add_argument("--runs", nargs="*", default=[], help="Optional run IDs to run.")
    parser.add_argument("--include-disabled", action="store_true", help="Run entries marked enabled: false.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--skip-run", action="store_true", help="Only score/aggregate existing result files.")
    parser.add_argument("--skip-score", action="store_true", help="Only run tasks; do not score or aggregate.")
    parser.add_argument("--resume", action="store_true", help="Resume existing runs.")
    parser.add_argument("--fresh", action="store_true", help="Start each run with --no-resume.")
    parser.add_argument("--limit", type=int, default=None, help="Override suite/group limit.")
    parser.add_argument("--sample", type=int, default=None, help="Override suite/group sample.")
    parser.add_argument("--seed", type=int, default=None, help="Override suite/group seed.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child scripts.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for experiment suite YAML files") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"suite must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for generated configs") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def deep_merge(base: Any, patch: Any) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        merged = dict(base)
        for key, value in patch.items():
            merged[key] = deep_merge(merged.get(key), value)
        return merged
    return copy.deepcopy(patch)


def resolve_model(run: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    ref = run.get("model_ref")
    if not ref:
        return {}
    model = models.get(str(ref))
    if not isinstance(model, dict):
        raise ValueError(f"run {run.get('id')} references unknown model_ref: {ref}")
    return model


def materialize_config(
    run: dict[str, Any],
    model_info: dict[str, Any],
    generated_dir: Path,
) -> Path:
    config_path = Path(str(run["config"]))
    base_config = load_yaml(config_path)
    merged = copy.deepcopy(base_config)

    if model_info.get("model"):
        merged["model"] = str(model_info["model"])
    if model_info.get("settings"):
        settings_path = generated_dir / "settings" / f"{run.get('model_ref', run['id'])}.yaml"
        write_yaml(settings_path, model_info["settings"])
        merged["settings_path"] = str(settings_path)
    elif model_info.get("settings_path"):
        merged["settings_path"] = str(model_info["settings_path"])

    overrides = run.get("config_overrides") or {}
    if overrides:
        merged = deep_merge(merged, overrides)

    out = generated_dir / f"{run['id']}.yaml"
    write_yaml(out, merged)
    return out


def shell_join(cmd: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in cmd)


def run_cmd(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print(shell_join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def selected_groups(suite: dict[str, Any], names: set[str]) -> list[dict[str, Any]]:
    groups = suite.get("groups") or []
    if not isinstance(groups, list):
        raise ValueError("suite groups must be a list")
    if not names:
        return groups
    return [group for group in groups if str(group.get("name")) in names]


def run_enabled(run: dict[str, Any], include_disabled: bool) -> bool:
    if include_disabled:
        return True
    return bool(run.get("enabled", True))


def main() -> None:
    args = parse_args()
    suite = load_yaml(args.suite)
    defaults = suite.get("defaults") or {}
    models = suite.get("models") or {}
    if not isinstance(models, dict):
        raise ValueError("suite models must be a mapping")

    group_filter = set(args.groups)
    run_filter = set(args.runs)
    groups = selected_groups(suite, group_filter)
    if group_filter and not groups:
        raise ValueError(f"no groups matched: {sorted(group_filter)}")

    generated_root = Path(str(defaults.get("generated_config_dir", "outputs/experiment_suites/generated_configs")))
    score_paths_by_group: dict[str, list[str]] = {}
    all_score_paths: list[str] = []

    for group in groups:
        group_name = str(group.get("name") or "group")
        runs = group.get("runs") or []
        if not isinstance(runs, list):
            raise ValueError(f"group {group_name} runs must be a list")
        group_score_paths: list[str] = []

        for run in runs:
            run_id = str(run.get("id") or "")
            if not run_id:
                raise ValueError(f"group {group_name} has a run without id")
            if run_filter and run_id not in run_filter:
                continue
            if not run_enabled(run, args.include_disabled):
                print(f"SKIP disabled run: {run_id}", flush=True)
                continue

            dataset = str(run.get("dataset") or group.get("dataset") or defaults.get("dataset"))
            gold = str(run.get("gold") or group.get("gold") or defaults.get("gold"))
            output_dir = Path(str(run.get("output_dir") or group.get("output_dir") or defaults.get("output_dir", "outputs/runs")))
            scores_dir = Path(str(run.get("scores_dir") or group.get("scores_dir") or defaults.get("scores_dir", "outputs/scores")))
            tables_dir = Path(str(run.get("tables_dir") or group.get("tables_dir") or defaults.get("tables_dir", "outputs/tables")))
            limit = args.limit if args.limit is not None else int(run.get("limit") or group.get("limit") or defaults.get("limit") or 0)
            sample = args.sample if args.sample is not None else int(run.get("sample") or group.get("sample") or defaults.get("sample") or 0)
            seed = args.seed if args.seed is not None else int(run.get("seed") or group.get("seed") or defaults.get("seed") or 20260519)
            no_progress = bool(run.get("no_progress", group.get("no_progress", defaults.get("no_progress", False))))
            resume = bool(run.get("resume", group.get("resume", defaults.get("resume", True))))
            if args.resume:
                resume = True
            if args.fresh:
                resume = False

            model_info = resolve_model(run, models)
            generated_dir = generated_root / group_name
            config_path = materialize_config(run, model_info, generated_dir)
            env = os.environ.copy()
            for key, value in (run.get("env") or {}).items():
                env[str(key)] = str(value)

            if not args.skip_run:
                cmd = [
                    args.python,
                    "experiments/run_raes.py",
                    "--config",
                    str(config_path),
                    "--dataset",
                    dataset,
                    "--output-dir",
                    str(output_dir),
                    "--run-id",
                    run_id,
                    "--seed",
                    str(seed),
                ]
                if limit:
                    cmd.extend(["--limit", str(limit)])
                if sample:
                    cmd.extend(["--sample", str(sample)])
                if not resume:
                    cmd.append("--no-resume")
                if no_progress:
                    cmd.append("--no-progress")
                run_cmd(cmd, env=env, dry_run=args.dry_run)

            if args.skip_score:
                continue

            score_path = scores_dir / f"{run_id}.scores.jsonl"
            group_score_paths.append(str(score_path))
            all_score_paths.append(str(score_path))
            score_cmd = [
                args.python,
                "experiments/score_raes.py",
                "--gold",
                gold,
                "--results",
                str(output_dir / run_id / "results.jsonl"),
                "--output",
                str(score_path),
            ]
            run_cmd(score_cmd, env=env, dry_run=args.dry_run)

        if group_score_paths:
            score_paths_by_group[group_name] = group_score_paths
            tables_dir = Path(str(group.get("tables_dir") or defaults.get("tables_dir", "outputs/tables")))
            aggregate_cmd = [
                args.python,
                "experiments/aggregate_results.py",
                "--scores",
                *group_score_paths,
                "--output-json",
                str(tables_dir / f"{group_name}.summary.json"),
                "--output-md",
                str(tables_dir / f"{group_name}.summary.md"),
            ]
            run_cmd(aggregate_cmd, env=os.environ.copy(), dry_run=args.dry_run)

    if all_score_paths and not args.skip_score:
        tables_dir = Path(str(defaults.get("tables_dir", "outputs/tables")))
        aggregate_cmd = [
            args.python,
            "experiments/aggregate_results.py",
            "--scores",
            *all_score_paths,
            "--output-json",
            str(tables_dir / "all_groups.summary.json"),
            "--output-md",
            str(tables_dir / "all_groups.summary.md"),
        ]
        run_cmd(aggregate_cmd, env=os.environ.copy(), dry_run=args.dry_run)

        manifest = {
            "suite": suite.get("suite_name"),
            "groups": score_paths_by_group,
            "all_scores": all_score_paths,
        }
        manifest_path = tables_dir / "score_manifest.json"
        print(f"write {manifest_path}", flush=True)
        if not args.dry_run:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
