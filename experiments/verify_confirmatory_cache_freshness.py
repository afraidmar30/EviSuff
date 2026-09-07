#!/usr/bin/env python3
"""Refuse stale confirmatory caches while allowing compatible run resumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--run-cache", nargs="+", required=True, metavar="RUN_ID=CACHE_DIR"
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def nonempty_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [item for item in path.rglob("*") if item.is_file() and item.stat().st_size]


def verify(run_root: Path, mappings: list[str]) -> dict:
    rows = []
    errors = []
    seen = set()
    for mapping in mappings:
        if "=" not in mapping:
            errors.append(f"invalid mapping: {mapping}")
            continue
        run_id, raw_cache = mapping.split("=", 1)
        if not run_id or run_id in seen:
            errors.append(f"missing or duplicate run ID: {run_id!r}")
            continue
        seen.add(run_id)
        cache_dir = Path(raw_cache)
        cache_files = nonempty_files(cache_dir)
        trajectory_dir = run_root / run_id / "trajectories"
        trajectories = [
            path for path in trajectory_dir.glob("*.trajectory.jsonl")
            if path.is_file() and path.stat().st_size
        ]
        allowed = not cache_files or bool(trajectories)
        if not allowed:
            errors.append(
                f"{run_id}: {len(cache_files)} cached files but no confirmatory trajectory"
            )
        rows.append({
            "run_id": run_id,
            "cache_dir": str(cache_dir),
            "cache_files": len(cache_files),
            "confirmatory_trajectories": len(trajectories),
            "state": (
                "fresh" if not cache_files
                else "resumable" if trajectories
                else "stale_cache_without_run"
            ),
            "passed": allowed,
        })
    return {
        "status": "passed" if not errors and len(rows) == len(mappings) else "failed",
        "run_root": str(run_root),
        "runs": rows,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    report = verify(Path(args.run_root), args.run_cache)
    rendered = json.dumps(report, indent=2)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
