#!/usr/bin/env python3
"""Merge agreements and prepare disagreement-only adjudication CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


OUTPUT_LABEL_FIELDS = (
    "source_policy_compliant",
    "conflict_disclosure_correct",
    "uncertainty_present",
    "uncertainty_appropriate",
    "stop_decision",
)
CLAIM_LABELS = {"supported", "partially_supported", "unsupported", "unclear"}
OUTPUT_LABELS = {
    "source_policy_compliant": {"yes", "no"},
    "conflict_disclosure_correct": {"yes", "no", "not_applicable"},
    "uncertainty_present": {"yes", "no"},
    "uncertainty_appropriate": {"yes", "no"},
    "stop_decision": {"appropriate", "premature", "unnecessary_search"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a-claims", required=True)
    parser.add_argument("--annotator-b-claims", required=True)
    parser.add_argument("--annotator-a-outputs", required=True)
    parser.add_argument("--annotator-b-outputs", required=True)
    parser.add_argument("--output-claims", required=True)
    parser.add_argument("--output-outputs", required=True)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args()


def read_csv(path: Path, key: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            item_id = str(row.get(key) or "").strip()
            if not item_id or item_id in rows:
                raise ValueError(f"{path}: missing or duplicate {key}={item_id!r}")
            rows[item_id] = row
    return fields, rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def adjudication_work_present(path: Path, label_fields: tuple[str, ...]) -> bool:
    """Protect an already started adjudication sheet from regeneration."""
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return True
    protected_fields = (*label_fields, "rationale")
    return any(
        str(row.get(field) or "").strip()
        for row in rows
        for field in protected_fields
    )


def main() -> None:
    args = parse_args()
    protected = []
    claim_output = Path(args.output_claims)
    output_output = Path(args.output_outputs)
    if adjudication_work_present(claim_output, ("support_label",)):
        protected.append(str(claim_output))
    if adjudication_work_present(output_output, OUTPUT_LABEL_FIELDS):
        protected.append(str(output_output))
    if protected:
        raise SystemExit(
            "Refusing to overwrite an existing adjudication sheet: "
            + ", ".join(protected)
            + ". Archive the sheets before deliberately regenerating them."
        )
    claim_fields, a_claims = read_csv(Path(args.annotator_a_claims), "blind_pair_id")
    _, b_claims = read_csv(Path(args.annotator_b_claims), "blind_pair_id")
    output_fields, a_outputs = read_csv(Path(args.annotator_a_outputs), "blind_output_id")
    _, b_outputs = read_csv(Path(args.annotator_b_outputs), "blind_output_id")
    if set(a_claims) != set(b_claims):
        raise ValueError("Claim annotation sheets have different IDs")
    if set(a_outputs) != set(b_outputs):
        raise ValueError("Output annotation sheets have different IDs")

    adjudicated_claims: list[dict[str, Any]] = []
    claim_disagreements = 0
    for item_id in sorted(a_claims):
        left, right = a_claims[item_id], b_claims[item_id]
        left_label, right_label = normalized(left.get("support_label")), normalized(right.get("support_label"))
        if left_label not in CLAIM_LABELS or right_label not in CLAIM_LABELS:
            raise ValueError(
                f"Claim {item_id}: invalid labels before adjudication: "
                f"A={left_label!r}, B={right_label!r}"
            )
        row = dict(left)
        row["annotator_a_label"] = left_label
        row["annotator_b_label"] = right_label
        row["annotator_a_rationale"] = str(left.get("rationale") or "")
        row["annotator_b_rationale"] = str(right.get("rationale") or "")
        row["support_label"] = left_label if left_label == right_label else ""
        row["rationale"] = "agreement" if left_label == right_label else ""
        claim_disagreements += left_label != right_label
        adjudicated_claims.append(row)

    adjudicated_outputs: list[dict[str, Any]] = []
    output_disagreements = Counter()
    for item_id in sorted(a_outputs):
        left, right = a_outputs[item_id], b_outputs[item_id]
        row = dict(left)
        for field in OUTPUT_LABEL_FIELDS:
            left_label, right_label = normalized(left.get(field)), normalized(right.get(field))
            if left_label not in OUTPUT_LABELS[field] or right_label not in OUTPUT_LABELS[field]:
                raise ValueError(
                    f"Output {item_id}: invalid {field} labels before adjudication: "
                    f"A={left_label!r}, B={right_label!r}"
                )
            row[f"annotator_a_{field}"] = left_label
            row[f"annotator_b_{field}"] = right_label
            row[field] = left_label if left_label == right_label else ""
            output_disagreements[field] += left_label != right_label
        row["annotator_a_rationale"] = str(left.get("rationale") or "")
        row["annotator_b_rationale"] = str(right.get("rationale") or "")
        row["rationale"] = "agreement" if not any(
            not row[field] for field in OUTPUT_LABEL_FIELDS
        ) else ""
        adjudicated_outputs.append(row)

    claim_out_fields = claim_fields + [
        field for field in (
            "annotator_a_label", "annotator_b_label",
            "annotator_a_rationale", "annotator_b_rationale",
        ) if field not in claim_fields
    ]
    output_out_fields = output_fields + [
        f"annotator_{annotator}_{field}"
        for field in OUTPUT_LABEL_FIELDS for annotator in ("a", "b")
        if f"annotator_{annotator}_{field}" not in output_fields
    ] + [
        field for field in ("annotator_a_rationale", "annotator_b_rationale")
        if field not in output_fields
    ]
    write_csv(claim_output, claim_out_fields, adjudicated_claims)
    write_csv(output_output, output_out_fields, adjudicated_outputs)
    report = {
        "claim_pairs": len(adjudicated_claims),
        "claim_disagreements": claim_disagreements,
        "claim_disagreement_rate": claim_disagreements / len(adjudicated_claims) if adjudicated_claims else None,
        "outputs": len(adjudicated_outputs),
        "output_disagreements": dict(output_disagreements),
        "adjudication_complete": claim_disagreements == 0 and not any(output_disagreements.values()),
        "note": "Blank fields in output CSVs require adjudication; agreed fields are prefilled.",
    }
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
