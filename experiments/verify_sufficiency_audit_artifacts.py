#!/usr/bin/env python3
"""Verify the complete four-system sufficiency-audit artifact chain."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GROUP_MAP = {
    "answerable": "answerable",
    "answerable_with_uncertainty": "uncertainty",
    "conflicting_no_resolution": "conflict",
    "not_enough_public_evidence": "insufficient-evidence",
}
EXPECTED_GROUPS = {
    "answerable": 7,
    "uncertainty": 6,
    "conflict": 6,
    "insufficient-evidence": 6,
}
SYSTEM_RUNS = {
    "base": "audit25_qwen3_base",
    "answer-sft": "audit25_qwen3_answer_sft",
    "no-gate": "audit25_qwen3_no_gate",
    "full-evisuff": "audit25_qwen3_full_evisuff",
}
CLAIM_LABELS = {"supported", "partially_supported", "unsupported", "unclear"}
OUTPUT_LABELS = {
    "source_policy_compliant": {"yes", "no"},
    "conflict_disclosure_correct": {"yes", "no", "not_applicable"},
    "uncertainty_present": {"yes", "no"},
    "uncertainty_appropriate": {"yes", "no"},
    "stop_decision": {"appropriate", "premature", "unnecessary_search"},
}
OUTPUT_CONTEXT_FIELDS = {
    "required_facets",
    "source_policy",
    "minimum_evidence_policy",
    "gold_source_inventory",
    "known_conflicts",
    "stop_condition",
}
CLAIM_SHEET_CONTENT_FIELDS = {
    "blind_output_id", "question", "claim", "citation_url",
    "citation_title", "citation_snippet", "source_type",
}
OUTPUT_SHEET_CONTENT_FIELDS = {
    "group", "question", "final_answer", "gold_final_answer",
    "uncertainty_requirement", "boundary_conditions", "required_facets",
    "evidence_points", "source_policy", "minimum_evidence_policy",
    "gold_source_inventory", "known_conflicts", "stop_condition",
    "search_calls", "fetch_calls", "total_tool_calls",
    "retrieved_source_count", "final_citation_count", "citation_inventory",
    "action_trace", "gate_trace",
}
LATEX_SYSTEM_NAMES = {
    "Base": "base",
    "Answer-SFT": "answer-sft",
    "No-gate": "no-gate",
    "Full EviSuff": "full-evisuff",
}
LATEX_CONDITION_NAMES = {
    "Answerable": "answerable",
    "Uncertainty": "uncertainty",
    "Conflict": "conflict",
    "Insufficient": "insufficient-evidence",
}
LATEX_METRICS = (
    "claim_support_rate",
    "source_policy_compliance",
    "conflict_disclosure_accuracy",
    "uncertainty_calibration_accuracy",
    "stop_decision_accuracy",
    "premature_finalization_rate",
    "unnecessary_search_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--audit-dir",
        default="outputs/audits/sufficiency_audit25_four_systems",
    )
    parser.add_argument("--require-human-labels", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
    return rows


def iter_jsonl(path: Path):
    """Stream large JSONL files without retaining full training corpora."""
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield line_no, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def load_interface_bundle(path: Path) -> dict[str, Any]:
    html = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="bundle" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"Missing embedded annotation bundle in {path}")
    bundle = json.loads(match.group(1))
    if not isinstance(bundle, dict):
        raise ValueError(f"Invalid embedded annotation bundle in {path}")
    return bundle


def rid(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("id") or "")


def norm(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def csv_text(value: Any) -> str:
    return "" if value is None else str(value)


def sheet_content_mismatches(
    master_rows: list[dict[str, Any]], sheet_rows: list[dict[str, str]],
    key: str, fields: set[str],
) -> list[dict[str, Any]]:
    """Find accidental edits or ID drift in a blinded annotation sheet."""
    master = {str(row.get(key) or ""): row for row in master_rows}
    sheet = {str(row.get(key) or ""): row for row in sheet_rows}
    mismatches: list[dict[str, Any]] = []
    for item_id in sorted(set(master) | set(sheet)):
        if item_id not in master or item_id not in sheet:
            mismatches.append({"id": item_id, "field": key, "reason": "ID set differs"})
            continue
        for field in sorted(fields):
            expected = csv_text(master[item_id].get(field))
            observed = csv_text(sheet[item_id].get(field))
            if observed != expected:
                mismatches.append({
                    "id": item_id,
                    "field": field,
                    "expected": expected[:160],
                    "observed": observed[:160],
                })
                if len(mismatches) >= 20:
                    return mismatches
    return mismatches


def conflict_label_matches_group(label: Any, group: str) -> bool:
    value = norm(label)
    return value in {"yes", "no"} if group == "conflict" else value == "not_applicable"


def validate_core_latex_table(
    text: str, report: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Validate the 16-cell paper table and optionally match it to report.json."""
    required_tokens = {
        "table_start": r"\begin{table*}",
        "table_end": r"\end{table*}",
        "caption": r"\caption{Direct evidence-sufficiency audit",
        "label": r"\label{tab:direct-sufficiency}",
        "csr_up": r"CSR$\uparrow$",
        "spc_up": r"SPC$\uparrow$",
        "cda_up": r"CDA$\uparrow$",
        "ucal_up": r"U-Cal$\uparrow$",
        "stop_up": r"Stop$\uparrow$",
        "prem_down": r"Prem.$\downarrow$",
        "unnec_down": r"Unnec.$\downarrow$",
    }
    missing_tokens = [name for name, token in required_tokens.items() if token not in text]
    system_pattern = "|".join(re.escape(name) for name in LATEX_SYSTEM_NAMES)
    condition_pattern = "|".join(re.escape(name) for name in LATEX_CONDITION_NAMES)
    row_pattern = re.compile(
        rf"^\s*({system_pattern})\s*&\s*({condition_pattern})\s*&\s*"
        rf"([^&]+?)\s*&\s*([^&]+?)\s*&\s*([^&]+?)\s*&\s*([^&]+?)\s*&\s*"
        rf"([^&]+?)\s*&\s*([^&]+?)\s*&\s*([^&]+?)\s*\\\\\s*$"
    )
    candidate_rows = [
        line for line in text.splitlines()
        if any(re.match(rf"^\s*{re.escape(name)}\s*&", line) for name in LATEX_SYSTEM_NAMES)
    ]
    parsed: dict[tuple[str, str], list[str]] = {}
    duplicates: list[str] = []
    malformed: list[str] = []
    invalid_cells: list[dict[str, Any]] = []
    report_mismatches: list[dict[str, Any]] = []
    for line in candidate_rows:
        match = row_pattern.match(line)
        if not match:
            malformed.append(line[:240])
            continue
        display_system, display_condition, *raw_values = match.groups()
        system = LATEX_SYSTEM_NAMES[display_system]
        condition = LATEX_CONDITION_NAMES[display_condition]
        key = (system, condition)
        if key in parsed:
            duplicates.append(f"{system}/{condition}")
            continue
        values = [value.strip() for value in raw_values]
        parsed[key] = values
        for metric, value in zip(LATEX_METRICS, values):
            numeric = re.fullmatch(r"(?:100(?:\.0)?|\d{1,2}(?:\.\d+)?)", value) is not None
            dash_allowed = metric == "conflict_disclosure_accuracy" and condition != "conflict"
            if not numeric and not (dash_allowed and value == "--"):
                invalid_cells.append({
                    "system": system, "condition": condition,
                    "metric": metric, "value": value,
                })
        if report is not None:
            metrics = (((report.get("direct") or {}).get(system) or {}).get(condition) or {}).get("metrics") or {}
            for metric, observed in zip(LATEX_METRICS, values):
                estimate = (metrics.get(metric) or {}).get("estimate")
                expected = "--" if estimate is None else f"{100 * float(estimate):.1f}"
                if observed != expected:
                    report_mismatches.append({
                        "system": system, "condition": condition,
                        "metric": metric, "observed": observed, "expected": expected,
                    })
    expected_cells = {
        (system, condition)
        for system in LATEX_SYSTEM_NAMES.values()
        for condition in LATEX_CONDITION_NAMES.values()
    }
    missing_cells = [f"{system}/{condition}" for system, condition in sorted(expected_cells - set(parsed))]
    extra_cells = [f"{system}/{condition}" for system, condition in sorted(set(parsed) - expected_cells)]
    detail = {
        "data_rows": len(candidate_rows),
        "parsed_unique_cells": len(parsed),
        "missing_tokens": missing_tokens,
        "missing_cells": missing_cells,
        "extra_cells": extra_cells,
        "duplicates": duplicates,
        "malformed_rows": malformed,
        "invalid_cells": invalid_cells[:20],
        "report_mismatches": report_mismatches[:20],
    }
    passed = (
        len(candidate_rows) == 16
        and len(parsed) == 16
        and not any((missing_tokens, missing_cells, extra_cells, duplicates, malformed,
                     invalid_cells, report_mismatches))
    )
    return passed, detail


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    audit_dir = (root / args.audit_dir).resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    question_path = root / "outputs/eval/raes_sufficiency_audit25_seed20260809.jsonl"
    gold_path = root / "outputs/eval/raes_sufficiency_audit25_seed20260809_gold.jsonl"
    selection_summary_path = root / "outputs/eval/raes_sufficiency_audit25_seed20260809_summary.json"
    questions = load_jsonl(question_path)
    gold_rows = load_jsonl(gold_path)
    selection_summary = json.loads(selection_summary_path.read_text(encoding="utf-8"))
    question_ids = [rid(row) for row in questions]
    gold = {rid(row): row for row in gold_rows}
    check("task_count", len(question_ids) == 25 and len(set(question_ids)) == 25, len(question_ids))
    check("gold_id_match", set(question_ids) == set(gold), {
        "questions": len(set(question_ids)), "gold": len(gold)
    })
    groups = Counter(GROUP_MAP.get(str(gold[sid].get("answerability") or ""), "unknown") for sid in question_ids)
    check("answerability_balance", dict(groups) == EXPECTED_GROUPS, dict(groups))
    splits = Counter(str(row.get("split") or "") for row in questions)
    check("heldout_splits_only", set(splits) <= {"test", "stress"}, dict(splits))
    expected_training_files = {
        "outputs/training_lzy/answer_sft_disjoint_test_stress/train.jsonl",
        "outputs/training_lzy/answer_sft_disjoint_test_stress/val.jsonl",
        "outputs/training_lzy/gate_learning_ablation_disjoint_questions/with_gate/train.jsonl",
        "outputs/training_lzy/gate_learning_ablation_disjoint_questions/with_gate/val.jsonl",
        "outputs/training_lzy/gate_learning_ablation_disjoint_questions/no_gate/train.jsonl",
        "outputs/training_lzy/gate_learning_ablation_disjoint_questions/no_gate/val.jsonl",
    }
    summary_training_files = set(selection_summary.get("training_message_files_scanned") or [])
    check("selection_summary_training_files", summary_training_files == expected_training_files, {
        "observed": sorted(summary_training_files),
        "expected": sorted(expected_training_files),
    })
    check("selection_summary_matches_tasks", (
        selection_summary.get("seed") == 20260809
        and selection_summary.get("task_count") == 25
        and selection_summary.get("selected_ids") == question_ids
        and selection_summary.get("selected_group_counts") == dict(groups)
        and int(selection_summary.get("training_overlap_excluded_count") or 0) > 0
    ), {
        "seed": selection_summary.get("seed"),
        "task_count": selection_summary.get("task_count"),
        "excluded": selection_summary.get("training_overlap_excluded_count"),
        "selected_ids_match": selection_summary.get("selected_ids") == question_ids,
        "selected_groups": selection_summary.get("selected_group_counts"),
    })

    audit_questions = {
        rid(row): norm_text(row.get("question")) for row in questions
    }
    training_paths = [
        root / "outputs/training_lzy/answer_sft_disjoint_test_stress/train.jsonl",
        root / "outputs/training_lzy/answer_sft_disjoint_test_stress/val.jsonl",
        root / "outputs/training_lzy/gate_learning_ablation_disjoint_questions/with_gate/train.jsonl",
        root / "outputs/training_lzy/gate_learning_ablation_disjoint_questions/with_gate/val.jsonl",
        root / "outputs/training_lzy/gate_learning_ablation_disjoint_questions/no_gate/train.jsonl",
        root / "outputs/training_lzy/gate_learning_ablation_disjoint_questions/no_gate/val.jsonl",
    ]
    training_overlaps: dict[str, list[dict[str, Any]]] = {}
    for path in training_paths:
        overlaps: list[dict[str, Any]] = []
        for line_no, row in iter_jsonl(path):
            user_text = norm_text("\n".join(
                str(message.get("content") or "")
                for message in row.get("messages") or []
                if message.get("role") == "user"
            ))
            for sample_id, question in audit_questions.items():
                if question and question in user_text:
                    overlaps.append({"sample_id": sample_id, "line": line_no})
        training_overlaps[str(path.relative_to(root))] = overlaps
    check(
        "no_exact_audit_question_in_any_train_or_val",
        not any(training_overlaps.values()),
        {path: values[:5] for path, values in training_overlaps.items()},
    )

    result_ids: dict[str, set[str]] = {}
    score_ids: dict[str, set[str]] = {}
    score_final_citation_counts: dict[tuple[str, str], int] = {}
    for system, run_id in SYSTEM_RUNS.items():
        result_path = root / f"outputs/eval/audit25_runs/{run_id}/results.jsonl"
        score_path = root / f"outputs/eval/audit25_scores/{run_id}.scores.jsonl"
        results = load_jsonl(result_path)
        scores = load_jsonl(score_path)
        ids = {rid(row) for row in results}
        sids = {rid(row) for row in scores}
        result_ids[system] = ids
        score_ids[system] = sids
        statuses = Counter(str(row.get("status") or "") for row in results)
        check(f"{system}:results", len(results) == 25 and ids == set(question_ids), {
            "rows": len(results), "unique_ids": len(ids)
        })
        check(f"{system}:success", statuses == {"success": 25}, dict(statuses))
        check(f"{system}:scores", len(scores) == 25 and sids == set(question_ids), {
            "rows": len(scores), "unique_ids": len(sids)
        })
        numeric_e = sum(
            isinstance(row.get("evidence_support"), (int, float)) for row in scores
        )
        check(f"{system}:numeric_E", numeric_e == 25, numeric_e)
        for row in scores:
            score_final_citation_counts[(system, rid(row))] = int(row.get("citation_count") or 0)
    check("four_system_common_results", all(ids == set(question_ids) for ids in result_ids.values()), {
        system: len(ids) for system, ids in result_ids.items()
    })

    output_master = load_jsonl(audit_dir / "outputs_master.jsonl")
    claim_master = load_jsonl(audit_dir / "claim_pairs_master.jsonl")
    output_ids = [str(row.get("output_id") or "") for row in output_master]
    per_system = Counter(str(row.get("system") or "") for row in output_master)
    per_group = Counter(str(row.get("group") or "") for row in output_master)
    check("audit_outputs", len(output_master) == 100 and len(set(output_ids)) == 100, len(output_master))
    check(
        "raw_outputs_preserved_visible_answers_sanitized",
        all("raw_final_answer" in row for row in output_master)
        and all("<think" not in str(row.get("final_answer") or "").casefold() for row in output_master),
        {
            "raw_preserved": sum("raw_final_answer" in row for row in output_master),
            "visible_answers_with_think": sum(
                "<think" in str(row.get("final_answer") or "").casefold()
                for row in output_master
            ),
        },
    )
    check(
        "compact_action_trace_covers_tool_calls",
        all(
            isinstance(row.get("action_trace"), list)
            and len(row.get("action_trace") or []) == int(row.get("total_tool_calls") or 0)
            for row in output_master
        ),
        {
            "mismatches": [
                {
                    "output_id": row.get("output_id"),
                    "trace_actions": len(row.get("action_trace") or [])
                    if isinstance(row.get("action_trace"), list) else None,
                    "total_tool_calls": row.get("total_tool_calls"),
                }
                for row in output_master
                if not isinstance(row.get("action_trace"), list)
                or len(row.get("action_trace") or []) != int(row.get("total_tool_calls") or 0)
            ][:8]
        },
    )
    check("audit_outputs_per_system", per_system == {system: 25 for system in SYSTEM_RUNS}, dict(per_system))
    check("audit_outputs_per_group", per_group == {group: count * 4 for group, count in EXPECTED_GROUPS.items()}, dict(per_group))
    visible_citation_alignment = all(
        int(row.get("final_citation_count") or 0)
        == score_final_citation_counts.get((str(row.get("system") or ""), str(row.get("sample_id") or "")))
        for row in output_master
    )
    check(
        "automatic_E_uses_visible_final_citations",
        visible_citation_alignment,
        {
            "mismatches": [
                {
                    "system": row.get("system"),
                    "sample_id": row.get("sample_id"),
                    "audit_final_citations": row.get("final_citation_count"),
                    "score_citations": score_final_citation_counts.get(
                        (str(row.get("system") or ""), str(row.get("sample_id") or ""))
                    ),
                }
                for row in output_master
                if int(row.get("final_citation_count") or 0)
                != score_final_citation_counts.get(
                    (str(row.get("system") or ""), str(row.get("sample_id") or ""))
                )
            ][:8]
        },
    )
    check("claim_pair_cap", 0 < len(claim_master) <= 500, len(claim_master))
    claim_pairs_per_system = Counter(str(row.get("system") or "") for row in claim_master)
    check(
        "claim_pairs_cover_all_systems",
        set(claim_pairs_per_system) == set(SYSTEM_RUNS)
        and all(claim_pairs_per_system[system] > 0 for system in SYSTEM_RUNS),
        dict(claim_pairs_per_system),
    )
    claim_pairs_per_system_group = Counter(
        (str(row.get("system") or ""), str(row.get("group") or ""))
        for row in claim_master
    )
    expected_system_groups = {
        (system, group) for system in SYSTEM_RUNS for group in EXPECTED_GROUPS
    }
    check(
        "claim_pairs_cover_every_system_group",
        all(claim_pairs_per_system_group[cell] > 0 for cell in expected_system_groups),
        {
            f"{system}/{group}": claim_pairs_per_system_group[(system, group)]
            for system, group in sorted(expected_system_groups)
        },
    )
    pairs_per_output = Counter(str(row.get("output_id") or "") for row in claim_master)
    check("pairs_per_output_cap", max(pairs_per_output.values(), default=0) <= 5, max(pairs_per_output.values(), default=0))
    check("claim_output_links", set(pairs_per_output) <= set(output_ids), len(set(pairs_per_output) - set(output_ids)))

    claim_sheet_orders: dict[str, list[str]] = {}
    output_sheet_orders: dict[str, list[str]] = {}
    for annotator in ("a", "b"):
        claim_fields, claim_rows = load_csv(audit_dir / f"annotator_{annotator}_claim_labels.csv")
        output_fields, output_rows = load_csv(audit_dir / f"annotator_{annotator}_output_labels.csv")
        claim_sheet_orders[annotator] = [str(row.get("blind_pair_id") or "") for row in claim_rows]
        output_sheet_orders[annotator] = [str(row.get("blind_output_id") or "") for row in output_rows]
        interface_ok = False
        interface_detail: Any = "not checked"
        try:
            interface = load_interface_bundle(
                audit_dir / f"annotator_{annotator}_interface.html"
            )
            interface_claims = (interface.get("claims") or {}).get("rows") or []
            interface_outputs = (interface.get("outputs") or {}).get("rows") or []
            interface_ok = (
                interface.get("annotator") == annotator
                and (interface.get("claims") or {}).get("fields") == claim_fields
                and (interface.get("outputs") or {}).get("fields") == output_fields
                and [str(row.get("blind_pair_id") or "") for row in interface_claims]
                == claim_sheet_orders[annotator]
                and [str(row.get("blind_output_id") or "") for row in interface_outputs]
                == output_sheet_orders[annotator]
                and not sheet_content_mismatches(
                    claim_master, interface_claims, "blind_pair_id", CLAIM_SHEET_CONTENT_FIELDS
                )
                and not sheet_content_mismatches(
                    output_master, interface_outputs, "blind_output_id", OUTPUT_SHEET_CONTENT_FIELDS
                )
            )
            interface_detail = {
                "annotator": interface.get("annotator"),
                "claims": len(interface_claims),
                "outputs": len(interface_outputs),
                "digest": interface.get("digest"),
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            interface_detail = str(exc)
        check(
            f"annotator_{annotator}:interface_matches_blinded_sheets",
            interface_ok,
            interface_detail,
        )
        check(f"annotator_{annotator}:claim_sheet_size", len(claim_rows) == len(claim_master), len(claim_rows))
        check(f"annotator_{annotator}:output_sheet_size", len(output_rows) == 100, len(output_rows))
        claim_content_mismatches = sheet_content_mismatches(
            claim_master, claim_rows, "blind_pair_id", CLAIM_SHEET_CONTENT_FIELDS
        )
        output_content_mismatches = sheet_content_mismatches(
            output_master, output_rows, "blind_output_id", OUTPUT_SHEET_CONTENT_FIELDS
        )
        check(
            f"annotator_{annotator}:claim_content_matches_master",
            not claim_content_mismatches,
            claim_content_mismatches,
        )
        check(
            f"annotator_{annotator}:output_content_matches_master",
            not output_content_mismatches,
            output_content_mismatches,
        )
        check(f"annotator_{annotator}:system_blind", "system" not in claim_fields and "system" not in output_fields, {
            "claim_fields": claim_fields, "output_fields": output_fields
        })
        check(
            f"annotator_{annotator}:raw_reasoning_hidden",
            "raw_final_answer" not in output_fields
            and all("<think" not in str(row.get("final_answer") or "").casefold() for row in output_rows),
            "raw field absent and visible answers contain no <think> blocks",
        )
        check(
            f"annotator_{annotator}:action_trace_present",
            {"action_trace", "total_tool_calls"} <= set(output_fields),
            sorted({"action_trace", "total_tool_calls"} - set(output_fields)),
        )
        check(
            f"annotator_{annotator}:decision_context_present",
            OUTPUT_CONTEXT_FIELDS <= set(output_fields)
            and all(
                str(row.get(field) or "").strip()
                for row in output_rows
                for field in OUTPUT_CONTEXT_FIELDS
            ),
            sorted(OUTPUT_CONTEXT_FIELDS - set(output_fields)),
        )
        check(
            f"annotator_{annotator}:claim_condition_blind",
            "group" not in claim_fields
            and "gold_answerability" not in claim_fields
            and "uncertainty_requirement" not in claim_fields,
            claim_fields,
        )
    check("independent_claim_order", claim_sheet_orders["a"] != claim_sheet_orders["b"], "different" if claim_sheet_orders["a"] != claim_sheet_orders["b"] else "same")
    check("independent_output_order", output_sheet_orders["a"] != output_sheet_orders["b"], "different" if output_sheet_orders["a"] != output_sheet_orders["b"] else "same")

    if args.require_human_labels:
        provenance_path = audit_dir / "HUMAN_ANNOTATION_PROVENANCE.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        annotators = provenance.get("annotators") or []
        distinct_ids = {str(row.get("anonymous_id") or "") for row in annotators if row.get("anonymous_id")}
        human_declarations = all(
            row.get("is_human") is True and row.get("independent_annotation") is True
            and str(row.get("completion_date") or "").strip()
            for row in annotators
        )
        adjudication_declared = bool(
            str(provenance.get("adjudicator_anonymous_id") or "").strip()
            and str(provenance.get("adjudication_completion_date") or "").strip()
        )
        check(
            "human_provenance",
            len(annotators) == 2
            and len(distinct_ids) == 2
            and human_declarations
            and adjudication_declared,
            provenance,
        )
        for annotator in ("a", "b"):
            _, claim_rows = load_csv(audit_dir / f"annotator_{annotator}_claim_labels.csv")
            _, output_rows = load_csv(audit_dir / f"annotator_{annotator}_output_labels.csv")
            valid_claims = all(norm(row.get("support_label")) in CLAIM_LABELS for row in claim_rows)
            valid_outputs = all(
                all(norm(row.get(field)) in allowed for field, allowed in OUTPUT_LABELS.items())
                for row in output_rows
            )
            output_groups = {
                str(row.get("blind_output_id") or ""): str(row.get("group") or "")
                for row in output_master
            }
            conflict_labels_consistent = all(
                conflict_label_matches_group(
                    row.get("conflict_disclosure_correct"),
                    output_groups.get(str(row.get("blind_output_id") or ""), ""),
                )
                for row in output_rows
            )
            check(f"annotator_{annotator}:complete_claim_labels", valid_claims, len(claim_rows))
            check(f"annotator_{annotator}:complete_output_labels", valid_outputs, len(output_rows))
            check(
                f"annotator_{annotator}:conflict_labels_match_condition",
                conflict_labels_consistent,
                "conflict=yes/no; all other conditions=not_applicable",
            )
        _, adjudicated_claims = load_csv(audit_dir / "adjudicated_claim_labels.csv")
        _, adjudicated_outputs = load_csv(audit_dir / "adjudicated_output_labels.csv")
        check("adjudicated_claims_complete", len(adjudicated_claims) == len(claim_master) and all(norm(row.get("support_label")) in CLAIM_LABELS for row in adjudicated_claims), len(adjudicated_claims))
        check("adjudicated_outputs_complete", len(adjudicated_outputs) == 100 and all(all(norm(row.get(field)) in allowed for field, allowed in OUTPUT_LABELS.items()) for row in adjudicated_outputs), len(adjudicated_outputs))
        check(
            "adjudicated_conflict_labels_match_condition",
            all(
                conflict_label_matches_group(
                    row.get("conflict_disclosure_correct"),
                    {
                        str(master.get("blind_output_id") or ""): str(master.get("group") or "")
                        for master in output_master
                    }.get(str(row.get("blind_output_id") or ""), ""),
                )
                for row in adjudicated_outputs
            ),
            "conflict=yes/no; all other conditions=not_applicable",
        )
        report_path = audit_dir / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        check("final_report_status", report.get("status") == "complete_two_annotator_adjudicated_audit", report.get("status"))
        check(
            "final_report_frozen_resampling",
            report.get("iterations") == 10000 and report.get("seed") == 20260809,
            {"iterations": report.get("iterations"), "seed": report.get("seed")},
        )
        report_systems = set(SYSTEM_RUNS)
        direct = report.get("direct") or {}
        expected_direct_groups = {"overall", *EXPECTED_GROUPS}
        base_metrics = {
            "claim_support_rate",
            "claim_support_rate_judgeable",
            "source_policy_compliance",
            "uncertainty_calibration_accuracy",
            "stop_decision_accuracy",
            "premature_finalization_rate",
            "unnecessary_search_rate",
        }
        direct_complete = set(direct) == report_systems
        direct_primary_intervals_complete = True
        for system in report_systems:
            groups_report = direct.get(system) or {}
            direct_complete = direct_complete and set(groups_report) == expected_direct_groups
            for group in expected_direct_groups:
                metrics = (groups_report.get(group) or {}).get("metrics") or {}
                direct_complete = direct_complete and base_metrics <= set(metrics)
                if group in {"overall", "conflict"}:
                    direct_complete = direct_complete and "conflict_disclosure_accuracy" in metrics
                if group == "overall":
                    direct_complete = direct_complete and {
                        "uncertainty_precision", "uncertainty_recall"
                    } <= set(metrics)
                required_nonmissing = {
                    "claim_support_rate",
                    "source_policy_compliance",
                    "uncertainty_calibration_accuracy",
                    "stop_decision_accuracy",
                    "premature_finalization_rate",
                    "unnecessary_search_rate",
                }
                if group in {"overall", "conflict"}:
                    required_nonmissing.add("conflict_disclosure_accuracy")
                direct_primary_intervals_complete = (
                    direct_primary_intervals_complete
                    and all(
                        (metrics.get(name) or {}).get(field) is not None
                        for name in required_nonmissing
                        for field in ("estimate", "ci_low", "ci_high", "n")
                    )
                )
        check("final_report_direct_metrics_complete", direct_complete, {
            system: sorted((direct.get(system) or {}).keys()) for system in report_systems
        })
        check(
            "final_report_primary_estimates_have_intervals",
            direct_primary_intervals_complete,
            "all required system/group estimates, 95% intervals, and denominators are non-missing",
        )

        claims_report = ((report.get("claims") or {}).get("by_system") or {})
        claim_proportions_complete = set(claims_report) == report_systems and all(
            set(((claims_report.get(system) or {}).get("proportions") or {})) == CLAIM_LABELS
            for system in report_systems
        )
        check("final_report_claim_proportions_complete", claim_proportions_complete, {
            system: sorted(((claims_report.get(system) or {}).get("proportions") or {}).keys())
            for system in report_systems
        })

        agreement = report.get("agreement") or {}
        expected_agreement = {"claim_support", *OUTPUT_LABELS}
        agreement_complete = expected_agreement <= set(agreement) and all(
            {"agreement", "agreement_ci_low", "agreement_ci_high", "kappa", "kappa_ci_low", "kappa_ci_high"}
            <= set(agreement.get(label) or {})
            for label in expected_agreement
        )
        check("final_report_agreement_with_intervals", agreement_complete, sorted(agreement))

        correlations = ((report.get("claims") or {}).get("correlation_with_automatic_E") or {})
        expected_correlations = report_systems | {"pooled"}
        correlations_complete = set(correlations) == expected_correlations and all(
            {"pearson", "spearman"} <= set(correlations.get(system) or {})
            for system in expected_correlations
        )
        check("final_report_human_E_correlations", correlations_complete, sorted(correlations))

        contrasts = report.get("paired_direct_contrasts") or {}
        expected_comparators = {"base", "answer-sft", "no-gate"}
        contrasts_complete = set(contrasts) == expected_comparators and all(
            set(contrasts.get(system) or {}) == expected_direct_groups
            for system in expected_comparators
        )
        check("final_report_paired_full_evisuff_contrasts", contrasts_complete, {
            system: sorted((contrasts.get(system) or {}).keys())
            for system in expected_comparators
        })
        check(
            "final_report_declared_scope",
            report.get("outputs") == 100
            and int(report.get("claim_source_pairs") or 0) == len(claim_master)
            and 0 < len(claim_master) <= 500,
            {
                "outputs": report.get("outputs"),
                "claim_source_pairs": report.get("claim_source_pairs"),
            },
        )
        core_table_path = audit_dir / "direct_table.tex"
        if core_table_path.is_file() and core_table_path.stat().st_size > 0:
            core_table_ok, core_table_detail = validate_core_latex_table(
                core_table_path.read_text(encoding="utf-8"), report
            )
        else:
            core_table_ok = False
            core_table_detail = {"path": str(core_table_path), "reason": "missing or empty"}
        check(
            "final_core_latex_table_complete_and_report_aligned",
            core_table_ok,
            core_table_detail,
        )

    passed = all(item["passed"] for item in checks)
    report = {
        "status": "passed" if passed else "failed",
        "require_human_labels": args.require_human_labels,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Sufficiency audit artifact verification", "", f"Status: **{report['status']}**", "",
             "| Check | Pass | Detail |", "|---|---:|---|"]
    for item in checks:
        detail = json.dumps(item["detail"], ensure_ascii=False).replace("|", "\\|")
        lines.append(f"| {item['name']} | {'yes' if item['passed'] else 'NO'} | `{detail}` |")
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "checks_passed", "checks_total")}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
