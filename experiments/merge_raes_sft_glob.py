#!/usr/bin/env python3
"""Merge exported RAES SFT directories selected by glob patterns."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.raes_eval.training_export import merge_sft_exports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", nargs="+", required=True, help="Glob(s) matching export dirs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="merged")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs: list[Path] = []
    for pattern in args.glob:
        dirs.extend(Path(path) for path in glob.glob(pattern))
    dirs = sorted({path for path in dirs if path.is_dir() and (path / "sft_messages.jsonl").exists()})
    if not dirs:
        raise SystemExit("No export directories with sft_messages.jsonl matched.")
    outputs = merge_sft_exports(export_dirs=dirs, output_dir=args.output_dir, name=args.name)
    print(f"merged_dirs: {len(dirs)}")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
