#!/usr/bin/env python3
"""Verify that the confirmatory four-system configs differ only by identity.

The comparison is intentionally strict: controller, tools, budget, decoding,
gate thresholds, and failure policy must match.  Only run/model/cache identity
and the corresponding local LLM endpoint may differ.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.raes_eval.runner import AgentConfig, load_config


EXPECTED = {
    "audit25_qwen3_base": ("openai/qwen3-base", "http://127.0.0.1:18011/v1"),
    "audit25_qwen3_answer_sft": ("openai/qwen3-answer", "http://127.0.0.1:18012/v1"),
    "audit25_qwen3_no_gate": ("openai/qwen3-no-gate", "http://127.0.0.1:18013/v1"),
    "audit25_qwen3_full_evisuff": ("openai/qwen3-with-gate", "http://127.0.0.1:18014/v1"),
}
TOP_IDENTITY = {"agent_name", "model", "settings_path", "cache_dir"}
LLM_IDENTITY = {"default_model", "side_query_model", "fallback_model", "base_url"}
EXPECTED_BUDGET = {
    "max_turns": 40,
    "max_search": 25,
    "max_fetch": 25,
    "academic_search": 10,
    "news_search": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output-json")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def normalized_top(config: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(config)
    for key in TOP_IDENTITY:
        payload.pop(key, None)
    return payload


def normalized_settings(settings: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(settings)
    llm = payload.get("llm")
    if not isinstance(llm, dict):
        raise ValueError("Settings are missing an llm mapping")
    for key in LLM_IDENTITY:
        llm.pop(key, None)
    return payload


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)
    configs: dict[str, dict[str, Any]] = {}
    settings: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for run_id, (expected_model, expected_endpoint) in EXPECTED.items():
        path = config_dir / f"{run_id}.yaml"
        if not path.is_file():
            errors.append(f"missing config: {path}")
            continue
        config = load_yaml(path)
        configs[run_id] = config
        if config.get("agent_name") != run_id:
            errors.append(f"{run_id}: agent_name={config.get('agent_name')!r}")
        if config.get("model") != expected_model:
            errors.append(f"{run_id}: model={config.get('model')!r}")
        settings_path = Path(str(config.get("settings_path") or ""))
        if not settings_path.is_file():
            errors.append(f"{run_id}: missing settings {settings_path}")
            continue
        current_settings = load_yaml(settings_path)
        settings[run_id] = current_settings
        llm = current_settings.get("llm") or {}
        if llm.get("base_url") != expected_endpoint:
            errors.append(f"{run_id}: base_url={llm.get('base_url')!r}")
        for key in ("default_model", "side_query_model", "fallback_model"):
            if llm.get(key) != expected_model:
                errors.append(f"{run_id}: {key}={llm.get(key)!r}")

        # Resolve the named preset plus any explicit override exactly as the
        # runner does. Equality across systems is not enough if all four share
        # the same wrong budget.
        resolved_budget = AgentConfig.from_dict(load_config(path)).budget.to_dict()
        if resolved_budget != EXPECTED_BUDGET:
            errors.append(
                f"{run_id}: resolved budget={resolved_budget!r}, "
                f"expected={EXPECTED_BUDGET!r}"
            )

    reference = "audit25_qwen3_base"
    if reference in configs:
        expected_top = normalized_top(configs[reference])
        for run_id, config in configs.items():
            if normalized_top(config) != expected_top:
                errors.append(f"{run_id}: non-identity top-level config differs from Base")
    if reference in settings:
        expected_settings = normalized_settings(settings[reference])
        for run_id, current_settings in settings.items():
            if normalized_settings(current_settings) != expected_settings:
                errors.append(f"{run_id}: non-identity model settings differ from Base")

    report = {
        "status": "passed" if not errors and len(configs) == 4 and len(settings) == 4 else "failed",
        "config_dir": str(config_dir),
        "systems_checked": sorted(configs),
        "allowed_top_level_differences": sorted(TOP_IDENTITY),
        "allowed_llm_setting_differences": sorted(LLM_IDENTITY),
        "required_resolved_budget": EXPECTED_BUDGET,
        "errors": errors,
    }
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
