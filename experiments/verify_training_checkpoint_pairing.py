#!/usr/bin/env python3
"""Verify update-matched with-gate/no-gate checkpoint training arguments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MATCH_FIELDS = (
    "model", "model_type", "template", "train_type", "torch_dtype",
    "max_length", "per_device_train_batch_size", "per_device_eval_batch_size",
    "gradient_accumulation_steps", "learning_rate", "weight_decay",
    "adam_beta1", "adam_beta2", "lr_scheduler_type", "warmup_ratio",
    "target_modules", "lora_rank", "lora_alpha", "lora_dropout",
    "sequence_parallel_size", "bf16", "gradient_checkpointing",
    "dataset_shuffle", "train_dataloader_shuffle", "seed", "data_seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-gate-checkpoint", required=True)
    parser.add_argument("--no-gate-checkpoint", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-steps", type=int, default=2058)
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        json.loads((checkpoint / "args.json").read_text(encoding="utf-8")),
        json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8")),
    )


def verify(
    with_checkpoint: Path, no_checkpoint: Path, expected_model: str,
    expected_steps: int, expected_seed: int,
) -> dict[str, Any]:
    with_args, with_state = load(with_checkpoint)
    no_args, no_state = load(no_checkpoint)
    mismatches = []
    for field in MATCH_FIELDS:
        if with_args.get(field) != no_args.get(field):
            mismatches.append({
                "field": field,
                "with_gate": with_args.get(field),
                "no_gate": no_args.get(field),
            })
    for name, args, state, split_token in (
        ("with_gate", with_args, with_state, "/with_gate/"),
        ("no_gate", no_args, no_state, "/no_gate/"),
    ):
        if args.get("model") != expected_model:
            mismatches.append({"field": f"{name}.model", "observed": args.get("model")})
        if args.get("seed") != expected_seed or args.get("data_seed") != expected_seed:
            mismatches.append({
                "field": f"{name}.seed",
                "seed": args.get("seed"),
                "data_seed": args.get("data_seed"),
            })
        dataset = " ".join(str(value) for value in args.get("dataset") or [])
        val_dataset = " ".join(str(value) for value in args.get("val_dataset") or [])
        if split_token not in dataset or split_token not in val_dataset:
            mismatches.append({
                "field": f"{name}.dataset_paths",
                "dataset": dataset,
                "val_dataset": val_dataset,
            })
        if (
            int(state.get("global_step", -1)) != expected_steps
            or int(state.get("max_steps", -1)) != expected_steps
        ):
            mismatches.append({
                "field": f"{name}.updates",
                "global_step": state.get("global_step"),
                "max_steps": state.get("max_steps"),
            })
    return {
        "status": "passed" if not mismatches else "failed",
        "comparison": "update-matched with-gate versus no-gate checkpoints",
        "with_gate_checkpoint": str(with_checkpoint),
        "no_gate_checkpoint": str(no_checkpoint),
        "expected_model": expected_model,
        "expected_steps": expected_steps,
        "expected_seed": expected_seed,
        "matched_fields": list(MATCH_FIELDS),
        "allowed_design_differences": [
            "dataset", "val_dataset", "num_train_epochs", "max_steps argument",
            "output_dir and run metadata",
        ],
        "mismatches": mismatches,
    }


def main() -> None:
    args = parse_args()
    report = verify(
        Path(args.with_gate_checkpoint),
        Path(args.no_gate_checkpoint),
        args.expected_model,
        args.expected_steps,
        args.expected_seed,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
