#!/usr/bin/env python3
"""Validate and summarize a two-annotator RAES sufficiency audit.

Confidence intervals resample model outputs (the independent unit), never
individual claims. Correlations with automatic E are calculated at output
level. The script refuses to publish incomplete or non-adjudicated labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable


CLAIM_LABELS = ("supported", "partially_supported", "unsupported", "unclear")
YES_NO = ("yes", "no")
CONFLICT_LABELS = ("yes", "no", "not_applicable")
STOP_LABELS = ("appropriate", "premature", "unnecessary_search")
OUTPUT_FIELDS = {
    "source_policy_compliant": YES_NO,
    "conflict_disclosure_correct": CONFLICT_LABELS,
    "uncertainty_present": YES_NO,
    "uncertainty_appropriate": YES_NO,
    "stop_decision": STOP_LABELS,
}
GROUPS = ("answerable", "uncertainty", "conflict", "insufficient-evidence")
SYSTEM_ORDER = ("base", "answer-sft", "no-gate", "full-evisuff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-claims", required=True)
    parser.add_argument("--master-outputs", required=True)
    parser.add_argument("--annotator-a-claims", required=True)
    parser.add_argument("--annotator-b-claims", required=True)
    parser.add_argument("--annotator-a-outputs", required=True)
    parser.add_argument("--annotator-b-outputs", required=True)
    parser.add_argument("--adjudicated-claims", required=True)
    parser.add_argument("--adjudicated-outputs", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-tex", help="Optional compact LaTeX core table.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
    return rows


def index_unique(rows: Iterable[dict[str, Any]], key: str, name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get(key) or "").strip()
        if not item_id:
            raise ValueError(f"{name}: missing {key}")
        if item_id in out:
            raise ValueError(f"{name}: duplicate {key}={item_id}")
        out[item_id] = row
    return out


def normalized(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def ordered_systems(outputs: dict[str, dict[str, Any]]) -> list[str]:
    present = {str(row["system"]) for row in outputs.values()}
    return [system for system in SYSTEM_ORDER if system in present] + sorted(
        present - set(SYSTEM_ORDER)
    )


def validate_labels(
    rows: dict[str, dict[str, Any]], field: str, allowed: Iterable[str], name: str
) -> None:
    allowed_set = set(allowed)
    invalid = {item_id: normalized(row.get(field)) for item_id, row in rows.items()
               if normalized(row.get(field)) not in allowed_set}
    if invalid:
        preview = list(invalid.items())[:8]
        raise ValueError(f"{name}: {len(invalid)} missing/invalid {field} labels; examples={preview}")


def require_same_ids(reference: set[str], candidate: set[str], name: str) -> None:
    if reference != candidate:
        missing = sorted(reference - candidate)[:5]
        extra = sorted(candidate - reference)[:5]
        raise ValueError(f"{name}: ID mismatch; missing={missing}, extra={extra}")


def cohen_kappa(left: list[str], right: list[str], labels: Iterable[str]) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        return {"n": len(left), "agreement": None, "kappa": None}
    categories = list(labels)
    agreement = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[label] / len(left)) * (right_counts[label] / len(right))
        for label in categories
    )
    kappa = None if math.isclose(expected, 1.0) else (agreement - expected) / (1.0 - expected)
    return {"n": len(left), "agreement": agreement, "kappa": kappa}


def agreement_bootstrap(
    item_ids: list[str],
    left: dict[str, str],
    right: dict[str, str],
    labels: Iterable[str],
    cluster_by_item: dict[str, str],
    iterations: int,
    rng: random.Random,
    cluster_unit: str,
) -> dict[str, Any]:
    """Bootstrap agreement without treating nested claims as independent."""
    items_by_cluster: dict[str, list[str]] = defaultdict(list)
    for item_id in item_ids:
        items_by_cluster[cluster_by_item[item_id]].append(item_id)
    cluster_ids = sorted(items_by_cluster)

    def calculate(ids: list[str]) -> dict[str, Any]:
        return cohen_kappa(
            [left[item_id] for item_id in ids],
            [right[item_id] for item_id in ids],
            labels,
        )

    observed = calculate(item_ids)
    agreement_draws: list[float] = []
    kappa_draws: list[float] = []
    for _ in range(iterations):
        sampled_clusters = [rng.choice(cluster_ids) for _ in cluster_ids]
        sampled_items = [
            item_id
            for cluster_id in sampled_clusters
            for item_id in items_by_cluster[cluster_id]
        ]
        draw = calculate(sampled_items)
        if draw["agreement"] is not None:
            agreement_draws.append(float(draw["agreement"]))
        if draw["kappa"] is not None:
            kappa_draws.append(float(draw["kappa"]))
    return {
        **observed,
        "clusters": len(cluster_ids),
        "cluster_unit": cluster_unit,
        "agreement_ci_low": percentile(agreement_draws, 0.025),
        "agreement_ci_high": percentile(agreement_draws, 0.975),
        "kappa_ci_low": percentile(kappa_draws, 0.025),
        "kappa_ci_high": percentile(kappa_draws, 0.975),
    }


def percentile(values: list[float], q: float) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    idx = (len(values) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(values) - 1)
    fraction = idx - lo
    return values[lo] * (1 - fraction) + values[hi] * fraction


def interval(observed: float | None, draws: list[float], n: int) -> dict[str, Any]:
    return {
        "n": n,
        "estimate": observed,
        "ci_low": percentile(draws, 0.025),
        "ci_high": percentile(draws, 0.975),
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor
        while end + 1 < len(order) and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end + 2) / 2
        for pos in range(cursor, end + 1):
            ranks[order[pos]] = rank
        cursor = end + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rankdata(xs), rankdata(ys)) if len(xs) >= 3 else None


def bootstrap_outputs(
    output_ids: list[str], statistic: Callable[[list[str]], float | None],
    iterations: int, rng: random.Random
) -> dict[str, Any]:
    observed = statistic(output_ids)
    draws: list[float] = []
    if output_ids:
        for _ in range(iterations):
            sampled = [rng.choice(output_ids) for _ in output_ids]
            value = statistic(sampled)
            if value is not None:
                draws.append(value)
    return interval(observed, draws, len(output_ids))


def claim_statistics(
    masters: dict[str, dict[str, Any]], labels: dict[str, dict[str, Any]],
    outputs: dict[str, dict[str, Any]], iterations: int, rng: random.Random
) -> tuple[dict[str, Any], dict[str, tuple[int, int, int]]]:
    pairs_by_output: dict[str, list[str]] = defaultdict(list)
    for pair_id, row in masters.items():
        pairs_by_output[str(row["output_id"])].append(pair_id)

    def proportions(output_ids: list[str]) -> dict[str, float | None]:
        counts = Counter()
        for output_id in output_ids:
            for pair_id in pairs_by_output[output_id]:
                counts[normalized(labels[pair_id]["support_label"])] += 1
        denominator = sum(counts.values())
        return {label: counts[label] / denominator if denominator else None for label in CLAIM_LABELS}

    by_system: dict[str, Any] = {}
    systems = ordered_systems(outputs)
    for system in systems:
        output_ids = [oid for oid, row in outputs.items()
                      if row["system"] == system and pairs_by_output.get(oid)]
        observed = proportions(output_ids)
        draws: dict[str, list[float]] = defaultdict(list)
        for _ in range(iterations):
            sampled = [rng.choice(output_ids) for _ in output_ids]
            draw = proportions(sampled)
            for label, value in draw.items():
                if value is not None:
                    draws[label].append(value)
        by_system[system] = {
            "outputs_with_claims": len(output_ids),
            "claim_pairs": sum(len(pairs_by_output[oid]) for oid in output_ids),
            "proportions": {
                label: interval(observed[label], draws[label], len(output_ids))
                for label in CLAIM_LABELS
            },
        }

    per_output_support: dict[str, float] = {}
    per_output_counts: dict[str, tuple[int, int, int]] = {}
    for output_id, pair_ids in pairs_by_output.items():
        pair_labels = [normalized(labels[pair_id]["support_label"]) for pair_id in pair_ids]
        judgeable = [label for label in pair_labels if label != "unclear"]
        if pair_labels:
            supported = sum(label == "supported" for label in pair_labels)
            per_output_counts[output_id] = (supported, len(pair_labels), len(judgeable))
            # Conservative primary rate: inaccessible/unclear evidence does
            # not disappear from the denominator.
            per_output_support[output_id] = supported / len(pair_labels)

    correlations: dict[str, Any] = {}
    for system in systems + ["pooled"]:
        ids = [oid for oid, support in per_output_support.items()
               if outputs[oid].get("automatic_E") not in (None, "")
               and (system == "pooled" or outputs[oid]["system"] == system)]

        def correlation(sampled: list[str], method: str) -> float | None:
            xs = [per_output_support[oid] for oid in sampled]
            ys = [float(outputs[oid]["automatic_E"]) for oid in sampled]
            return pearson(xs, ys) if method == "pearson" else spearman(xs, ys)

        if system == "pooled":
            by_task: dict[str, list[str]] = defaultdict(list)
            for oid in ids:
                by_task[str(outputs[oid]["sample_id"])].append(oid)
            clusters = sorted(by_task)

            def pooled_correlation(sampled_tasks: list[str], method: str) -> float | None:
                sampled_outputs = [oid for task in sampled_tasks for oid in by_task[task]]
                return correlation(sampled_outputs, method)

            correlations[system] = {
                method: bootstrap_outputs(
                    clusters,
                    lambda sampled, method=method: pooled_correlation(sampled, method),
                    iterations,
                    rng,
                )
                for method in ("pearson", "spearman")
            }
            correlations[system]["cluster_unit"] = "task"
        else:
            correlations[system] = {
                method: bootstrap_outputs(
                    ids, lambda sampled, method=method: correlation(sampled, method), iterations, rng
                )
                for method in ("pearson", "spearman")
            }
            correlations[system]["cluster_unit"] = "output/task"
    return {"by_system": by_system, "correlation_with_automatic_E": correlations}, per_output_counts


def binary_rate(
    rows: dict[str, dict[str, Any]], output_ids: list[str], field: str, positive: str
) -> float | None:
    values = [normalized(rows[oid][field]) == positive for oid in output_ids]
    return mean(values) if values else None


def precision_recall(rows: dict[str, dict[str, Any]], outputs: dict[str, dict[str, Any]], ids: list[str]) -> tuple[float | None, float | None]:
    truth = [outputs[oid]["group"] != "answerable" for oid in ids]
    pred = [normalized(rows[oid]["uncertainty_present"]) == "yes" for oid in ids]
    tp = sum(expected and observed for expected, observed in zip(truth, pred))
    fp = sum(not expected and observed for expected, observed in zip(truth, pred))
    fn = sum(expected and not observed for expected, observed in zip(truth, pred))
    return (tp / (tp + fp) if tp + fp else None, tp / (tp + fn) if tp + fn else None)


def direct_statistics(
    labels: dict[str, dict[str, Any]], outputs: dict[str, dict[str, Any]],
    claim_counts: dict[str, tuple[int, int, int]], iterations: int, rng: random.Random
) -> dict[str, Any]:
    systems = ordered_systems(outputs)
    report: dict[str, Any] = {}
    for system in systems:
        report[system] = {}
        system_ids = [oid for oid, row in outputs.items() if row["system"] == system]
        for group in ("overall",) + GROUPS:
            ids = system_ids if group == "overall" else [oid for oid in system_ids if outputs[oid]["group"] == group]
            metric_fns: dict[str, Callable[[list[str]], float | None]] = {
                "source_policy_compliance": lambda sample: binary_rate(labels, sample, "source_policy_compliant", "yes"),
                "uncertainty_calibration_accuracy": lambda sample: binary_rate(labels, sample, "uncertainty_appropriate", "yes"),
                "stop_decision_accuracy": lambda sample: binary_rate(labels, sample, "stop_decision", "appropriate"),
                "premature_finalization_rate": lambda sample: binary_rate(labels, sample, "stop_decision", "premature"),
                "unnecessary_search_rate": lambda sample: binary_rate(labels, sample, "stop_decision", "unnecessary_search"),
            }
            metrics = {
                name: bootstrap_outputs(ids, function, iterations, rng)
                for name, function in metric_fns.items()
            }
            if group in ("overall", "conflict"):
                conflict_ids = [oid for oid in ids if outputs[oid]["group"] == "conflict"]
                metrics["conflict_disclosure_accuracy"] = bootstrap_outputs(
                    conflict_ids,
                    lambda sample: binary_rate(
                        labels, sample, "conflict_disclosure_correct", "yes"
                    ),
                    iterations,
                    rng,
                )
            claim_ids = [oid for oid in ids if oid in claim_counts]

            def claim_support_rate(sample: list[str]) -> float | None:
                supported = sum(claim_counts[oid][0] for oid in sample)
                denominator = sum(claim_counts[oid][1] for oid in sample)
                return supported / denominator if denominator else None

            def claim_support_rate_judgeable(sample: list[str]) -> float | None:
                supported = sum(claim_counts[oid][0] for oid in sample)
                denominator = sum(claim_counts[oid][2] for oid in sample)
                return supported / denominator if denominator else None

            metrics["claim_support_rate"] = bootstrap_outputs(
                claim_ids, claim_support_rate, iterations, rng
            )
            metrics["claim_support_rate_judgeable"] = bootstrap_outputs(
                claim_ids, claim_support_rate_judgeable, iterations, rng
            )
            if group == "overall":
                for metric_index, metric_name in enumerate(("uncertainty_precision", "uncertainty_recall")):
                    metrics[metric_name] = bootstrap_outputs(
                        ids,
                        lambda sample, index=metric_index: precision_recall(labels, outputs, sample)[index],
                        iterations,
                        rng,
                    )
            report[system][group] = {"outputs": len(ids), "metrics": metrics}
    return report


def paired_direct_contrasts(
    labels: dict[str, dict[str, Any]], outputs: dict[str, dict[str, Any]],
    claim_counts: dict[str, tuple[int, int, int]], iterations: int, rng: random.Random,
    target_system: str = "full-evisuff",
) -> dict[str, Any]:
    """Estimate paired task-level differences against the target system.

    Every bootstrap draw resamples shared task identifiers and then includes
    the corresponding output from both systems.  This preserves the paired
    four-system design and avoids treating outputs for the same task as
    independent observations.
    """
    by_system_task = {
        (str(row["system"]), str(row["sample_id"])): output_id
        for output_id, row in outputs.items()
    }
    systems = ordered_systems(outputs)
    if target_system not in systems:
        raise ValueError(f"Missing target system for paired contrasts: {target_system}")

    def ids_for(system: str, task_ids: list[str]) -> list[str]:
        return [by_system_task[(system, task_id)] for task_id in task_ids]

    def metric_value(name: str, system: str, task_ids: list[str]) -> float | None:
        ids = ids_for(system, task_ids)
        if name == "claim_support_rate":
            usable = [output_id for output_id in ids if output_id in claim_counts]
            supported = sum(claim_counts[output_id][0] for output_id in usable)
            denominator = sum(claim_counts[output_id][1] for output_id in usable)
            return supported / denominator if denominator else None
        if name == "claim_support_rate_judgeable":
            usable = [output_id for output_id in ids if output_id in claim_counts]
            supported = sum(claim_counts[output_id][0] for output_id in usable)
            denominator = sum(claim_counts[output_id][2] for output_id in usable)
            return supported / denominator if denominator else None
        if name == "source_policy_compliance":
            return binary_rate(labels, ids, "source_policy_compliant", "yes")
        if name == "conflict_disclosure_accuracy":
            conflict_ids = [
                output_id for output_id in ids
                if outputs[output_id]["group"] == "conflict"
            ]
            return binary_rate(
                labels, conflict_ids, "conflict_disclosure_correct", "yes"
            )
        if name == "uncertainty_calibration_accuracy":
            return binary_rate(labels, ids, "uncertainty_appropriate", "yes")
        if name == "uncertainty_precision":
            return precision_recall(labels, outputs, ids)[0]
        if name == "uncertainty_recall":
            return precision_recall(labels, outputs, ids)[1]
        if name == "stop_decision_accuracy":
            return binary_rate(labels, ids, "stop_decision", "appropriate")
        if name == "premature_finalization_rate":
            return binary_rate(labels, ids, "stop_decision", "premature")
        if name == "unnecessary_search_rate":
            return binary_rate(labels, ids, "stop_decision", "unnecessary_search")
        raise KeyError(name)

    metric_names = (
        "claim_support_rate",
        "claim_support_rate_judgeable",
        "source_policy_compliance",
        "conflict_disclosure_accuracy",
        "uncertainty_calibration_accuracy",
        "uncertainty_precision",
        "uncertainty_recall",
        "stop_decision_accuracy",
        "premature_finalization_rate",
        "unnecessary_search_rate",
    )
    report: dict[str, Any] = {}
    for comparator in systems:
        if comparator == target_system:
            continue
        common_tasks = sorted(
            task_id
            for system, task_id in by_system_task
            if system == target_system and (comparator, task_id) in by_system_task
        )
        report[comparator] = {}
        for group in ("overall",) + GROUPS:
            task_ids = common_tasks if group == "overall" else [
                task_id for task_id in common_tasks
                if outputs[by_system_task[(target_system, task_id)]]["group"] == group
            ]
            metrics: dict[str, Any] = {}
            for name in metric_names:
                if name == "conflict_disclosure_accuracy" and group not in ("overall", "conflict"):
                    continue
                if name in ("uncertainty_precision", "uncertainty_recall") and group != "overall":
                    continue

                metric_task_ids = task_ids
                if name in ("claim_support_rate", "claim_support_rate_judgeable"):
                    metric_task_ids = [
                        task_id for task_id in task_ids
                        if by_system_task[(target_system, task_id)] in claim_counts
                        and by_system_task[(comparator, task_id)] in claim_counts
                    ]

                def difference(sampled_tasks: list[str], metric_name: str = name) -> float | None:
                    target = metric_value(metric_name, target_system, sampled_tasks)
                    reference = metric_value(metric_name, comparator, sampled_tasks)
                    if target is None or reference is None:
                        return None
                    return target - reference

                metrics[name] = bootstrap_outputs(
                    metric_task_ids, difference, iterations, rng
                )
            report[comparator][group] = {
                "tasks": len(task_ids),
                "claim_support_paired_tasks": (
                    metrics.get("claim_support_rate") or {}
                ).get("n", 0),
                "contrast": f"{target_system} minus {comparator}",
                "cluster_unit": "paired task",
                "metrics": metrics,
            }
    return report


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def fmt_ci(metric: dict[str, Any]) -> str:
    if metric["estimate"] is None:
        return "NA"
    return f"{fmt(metric['estimate'])} [{fmt(metric['ci_low'])}, {fmt(metric['ci_high'])}]"


def fmt_agreement_ci(values: dict[str, Any], field: str) -> str:
    estimate = values.get(field)
    if estimate is None:
        return "NA"
    return (
        f"{fmt(estimate)} [{fmt(values.get(field + '_ci_low'))}, "
        f"{fmt(values.get(field + '_ci_high'))}]"
    )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Claim-level and direct evidence-sufficiency audit", "",
        f"System-specific estimates resample outputs (one per task); pooled agreement and correlation resample tasks. Bootstrap replicates: {report['iterations']}.", "",
        "## Inter-annotator agreement", "",
        "| Label | N | Clusters | Raw agreement (95% CI) | Cohen's kappa (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in report["agreement"].items():
        lines.append(
            f"| {label} | {values['n']} | {values['clusters']} | "
            f"{fmt_agreement_ci(values, 'agreement')} | "
            f"{fmt_agreement_ci(values, 'kappa')} |"
        )
    lines += ["", "## Adjudicated claim support by system", "",
              "| System | Outputs | Pairs | Supported | Partial | Unsupported | Unclear |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for system, values in report["claims"]["by_system"].items():
        proportions = values["proportions"]
        lines.append(
            f"| {system} | {values['outputs_with_claims']} | {values['claim_pairs']} | "
            f"{fmt_ci(proportions['supported'])} | {fmt_ci(proportions['partially_supported'])} | "
            f"{fmt_ci(proportions['unsupported'])} | {fmt_ci(proportions['unclear'])} |"
        )
    lines += ["", "## Human support vs automatic E (output level)", "",
              "| System | N | Pearson r | Spearman rho |", "|---|---:|---:|---:|"]
    for system, values in report["claims"]["correlation_with_automatic_E"].items():
        lines.append(
            f"| {system} | {values['pearson']['n']} | {fmt_ci(values['pearson'])} | "
            f"{fmt_ci(values['spearman'])} |"
        )
    lines += ["", "## Direct sufficiency metrics", "",
              "Values are estimates with output-level bootstrap 95% CIs. Primary Claim support is strict `supported` among all audited pairs; CSR-judgeable excludes `unclear` pairs as a sensitivity analysis.", "",
              "| System | Group | N outputs | CSR N | Claim support | CSR-judgeable | Source policy | Conflict disclosure | Uncertainty calibration | Uncertainty P | Uncertainty R | Stop accuracy | Premature | Unnecessary search |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for system, groups in report["direct"].items():
        for group, values in groups.items():
            metrics = values["metrics"]
            get = lambda name: fmt_ci(metrics[name]) if name in metrics else "NA"
            lines.append(
                f"| {system} | {group} | {values['outputs']} | "
                f"{metrics['claim_support_rate']['n']} | {get('claim_support_rate')} | "
                f"{get('claim_support_rate_judgeable')} | "
                f"{get('source_policy_compliance')} | "
                f"{get('conflict_disclosure_accuracy')} | {get('uncertainty_calibration_accuracy')} | "
                f"{get('uncertainty_precision')} | {get('uncertainty_recall')} | "
                f"{get('stop_decision_accuracy')} | {get('premature_finalization_rate')} | "
                f"{get('unnecessary_search_rate')} |"
            )
    lines += [
        "", "## Paired Full EviSuff contrasts", "",
        "Values are paired task-bootstrap differences (Full EviSuff minus comparator). "
        "Positive values favor Full EviSuff for Claim support, Source policy, Conflict disclosure, "
        "Uncertainty calibration/precision/recall, and Stop accuracy; negative values favor Full "
        "EviSuff for Premature finalization and Unnecessary search.", "",
        "| Comparator | Group | N tasks | Paired CSR N | Claim support | CSR-judgeable | Source policy | Conflict disclosure | Uncertainty calibration | Uncertainty P | Uncertainty R | Stop accuracy | Premature | Unnecessary search |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comparator, groups in report["paired_direct_contrasts"].items():
        for group, values in groups.items():
            metrics = values["metrics"]
            get = lambda name: fmt_ci(metrics[name]) if name in metrics else "NA"
            lines.append(
                f"| {comparator} | {group} | {values['tasks']} | "
                f"{values['claim_support_paired_tasks']} | "
                f"{get('claim_support_rate')} | {get('claim_support_rate_judgeable')} | "
                f"{get('source_policy_compliance')} | "
                f"{get('conflict_disclosure_accuracy')} | "
                f"{get('uncertainty_calibration_accuracy')} | "
                f"{get('uncertainty_precision')} | {get('uncertainty_recall')} | "
                f"{get('stop_decision_accuracy')} | "
                f"{get('premature_finalization_rate')} | "
                f"{get('unnecessary_search_rate')} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(report: dict[str, Any], path: Path) -> None:
    group_names = {
        "answerable": "Answerable",
        "uncertainty": "Uncertainty",
        "conflict": "Conflict",
        "insufficient-evidence": "Insufficient",
    }
    system_names = {
        "base": "Base",
        "answer-sft": "Answer-SFT",
        "no-gate": "No-gate",
        "full-evisuff": "Full EviSuff",
    }

    def value(metrics: dict[str, Any], name: str) -> str:
        metric = metrics.get(name)
        if not metric or metric.get("estimate") is None:
            return "--"
        return f"{100 * float(metric['estimate']):.1f}"

    lines = [
        "% Generated by experiments/summarize_claim_support_audit.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Direct evidence-sufficiency audit (\\%; arrows show the favorable direction). CSR: adjudicated supported pairs divided by all audited pairs (unclear counts as non-support); SPC: source-policy compliance; CDA: conflict-disclosure accuracy; U-Cal: uncertainty calibration; Stop: stop-decision accuracy; Prem.: premature finalization; Unnec.: unnecessary search. Task-cluster 95\\% CIs and denominators are reported in the released audit report.}",
        "\\label{tab:direct-sufficiency}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.1pt}",
        "\\begin{tabular}{@{}llrrrrrrr@{}}",
        "\\toprule",
        "System & Condition & CSR$\\uparrow$ & SPC$\\uparrow$ & CDA$\\uparrow$ & U-Cal$\\uparrow$ & Stop$\\uparrow$ & Prem.$\\downarrow$ & Unnec.$\\downarrow$ \\\\",
        "\\midrule",
    ]
    systems = list(report["direct"])
    for system_index, system in enumerate(systems):
        for group in GROUPS:
            metrics = report["direct"][system][group]["metrics"]
            lines.append(
                f"{system_names.get(system, system)} & {group_names[group]} & "
                f"{value(metrics, 'claim_support_rate')} & "
                f"{value(metrics, 'source_policy_compliance')} & "
                f"{value(metrics, 'conflict_disclosure_accuracy')} & "
                f"{value(metrics, 'uncertainty_calibration_accuracy')} & "
                f"{value(metrics, 'stop_decision_accuracy')} & "
                f"{value(metrics, 'premature_finalization_rate')} & "
                f"{value(metrics, 'unnecessary_search_rate')} \\\\"
            )
        if system_index + 1 < len(systems):
            lines.append("\\addlinespace[1pt]")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.iterations < 100:
        raise SystemExit("Use at least 100 bootstrap iterations")
    rng = random.Random(args.seed)
    claim_master = index_unique(load_rows(Path(args.master_claims)), "blind_pair_id", "master claims")
    output_master_blind = index_unique(load_rows(Path(args.master_outputs)), "blind_output_id", "master outputs")
    output_master = {str(row["output_id"]): row for row in output_master_blind.values()}
    blind_to_output = {blind: str(row["output_id"]) for blind, row in output_master_blind.items()}

    a_claims = index_unique(load_rows(Path(args.annotator_a_claims)), "blind_pair_id", "annotator A claims")
    b_claims = index_unique(load_rows(Path(args.annotator_b_claims)), "blind_pair_id", "annotator B claims")
    final_claims = index_unique(load_rows(Path(args.adjudicated_claims)), "blind_pair_id", "adjudicated claims")
    for name, rows in (("annotator A claims", a_claims), ("annotator B claims", b_claims), ("adjudicated claims", final_claims)):
        require_same_ids(set(claim_master), set(rows), name)
        validate_labels(rows, "support_label", CLAIM_LABELS, name)

    a_outputs = index_unique(load_rows(Path(args.annotator_a_outputs)), "blind_output_id", "annotator A outputs")
    b_outputs = index_unique(load_rows(Path(args.annotator_b_outputs)), "blind_output_id", "annotator B outputs")
    final_outputs_blind = index_unique(load_rows(Path(args.adjudicated_outputs)), "blind_output_id", "adjudicated outputs")
    for name, rows in (("annotator A outputs", a_outputs), ("annotator B outputs", b_outputs), ("adjudicated outputs", final_outputs_blind)):
        require_same_ids(set(output_master_blind), set(rows), name)
        for field, allowed in OUTPUT_FIELDS.items():
            validate_labels(rows, field, allowed, name)
    final_outputs = {blind_to_output[blind]: row for blind, row in final_outputs_blind.items()}

    claim_ids = sorted(claim_master)
    output_ids = sorted(output_master_blind)
    agreement = {
        "claim_support": agreement_bootstrap(
            claim_ids,
            {item_id: normalized(a_claims[item_id]["support_label"]) for item_id in claim_ids},
            {item_id: normalized(b_claims[item_id]["support_label"]) for item_id in claim_ids},
            CLAIM_LABELS,
            {item_id: str(claim_master[item_id]["sample_id"]) for item_id in claim_ids},
            args.iterations,
            rng,
            "task",
        )
    }
    for field, allowed in OUTPUT_FIELDS.items():
        agreement[field] = agreement_bootstrap(
            output_ids,
            {item_id: normalized(a_outputs[item_id][field]) for item_id in output_ids},
            {item_id: normalized(b_outputs[item_id][field]) for item_id in output_ids},
            allowed,
            {item_id: str(output_master_blind[item_id]["sample_id"]) for item_id in output_ids},
            args.iterations,
            rng,
            "task",
        )

    # Master claim rows refer to raw output IDs; final labels remain keyed by blind pair ID.
    claims, claim_counts = claim_statistics(
        claim_master, final_claims, output_master, args.iterations, rng
    )
    direct = direct_statistics(final_outputs, output_master, claim_counts, args.iterations, rng)
    paired_contrasts = paired_direct_contrasts(
        final_outputs, output_master, claim_counts, args.iterations, rng
    )
    report = {
        "status": "complete_two_annotator_adjudicated_audit",
        "iterations": args.iterations,
        "seed": args.seed,
        "outputs": len(output_master),
        "claim_source_pairs": len(claim_master),
        "agreement": agreement,
        "claims": claims,
        "direct": direct,
        "paired_direct_contrasts": paired_contrasts,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.output_md))
    if args.output_tex:
        write_latex(report, Path(args.output_tex))
    print(output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
