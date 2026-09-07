#!/usr/bin/env python3
"""Create isolated, deterministic handoff archives for two human annotators."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from experiments.verify_sufficiency_audit_artifacts import load_interface_bundle


FORBIDDEN_FIELDS = {"system", "output_id", "raw_final_answer", "raw_reasoning"}
README = """EviSuff independent human audit — annotator {annotator}

1. Extract this archive and open evisuff_annotator_{annotator_lower}.html locally in a modern browser.
2. Complete every Claim-support item and every Output-level item. The interface autosaves in this browser.
3. Export the Claim CSV and Output CSV separately after both progress counters reach 100%.
4. Return only those two exported CSV files to the audit coordinator.
5. Do not discuss individual items or exchange files with the other annotator until both independent completion dates have been recorded.

The interface is self-contained and does not call an AI model. All judgments must be made by the named human annotator.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_handoff_bundle(bundle: dict[str, Any], annotator: str) -> dict[str, Any]:
    claim_fields = set((bundle.get("claims") or {}).get("fields") or [])
    output_fields = set((bundle.get("outputs") or {}).get("fields") or [])
    forbidden = sorted((claim_fields | output_fields) & FORBIDDEN_FIELDS)
    claims = (bundle.get("claims") or {}).get("rows") or []
    outputs = (bundle.get("outputs") or {}).get("rows") or []
    if bundle.get("annotator") != annotator:
        raise ValueError(
            f"Interface annotator mismatch: expected {annotator}, got {bundle.get('annotator')}"
        )
    if forbidden:
        raise ValueError(f"Identity/raw fields forbidden in handoff: {forbidden}")
    if not claims or not outputs:
        raise ValueError("Handoff interface must contain nonempty claim and output rows")
    if not bundle.get("digest"):
        raise ValueError("Handoff interface is missing its content-binding digest")
    return {
        "annotator": annotator,
        "claims": len(claims),
        "outputs": len(outputs),
        "interface_digest": str(bundle["digest"]),
    }


def deterministic_zip(path: Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, files[name])


def main() -> None:
    args = parse_args()
    audit_dir = Path(args.audit_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    preflight_path = audit_dir / "pre_annotation_verification.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed" or preflight.get("require_human_labels") is not False:
        raise ValueError(f"Pre-annotation verification did not pass: {preflight_path}")

    manifest: dict[str, Any] = {
        "status": "ready_for_two_independent_human_annotators",
        "pre_annotation_verification": str(preflight_path),
        "annotators": {},
    }
    seen_interface_digests: set[str] = set()
    for annotator in ("a", "b"):
        interface_path = audit_dir / f"annotator_{annotator}_interface.html"
        interface_bytes = interface_path.read_bytes()
        bundle = load_interface_bundle(interface_path)
        detail = validate_handoff_bundle(bundle, annotator)
        if detail["interface_digest"] in seen_interface_digests:
            raise ValueError("Annotator interfaces must have distinct content/order digests")
        seen_interface_digests.add(detail["interface_digest"])
        archive_path = output_dir / f"evisuff_human_audit_annotator_{annotator}.zip"
        interface_name = f"evisuff_annotator_{annotator}.html"
        deterministic_zip(archive_path, {
            interface_name: interface_bytes,
            "README.txt": README.format(
                annotator=annotator.upper(), annotator_lower=annotator
            ).encode("utf-8"),
        })
        manifest["annotators"][annotator] = {
            **detail,
            "archive": str(archive_path),
            "archive_sha256": sha256_bytes(archive_path.read_bytes()),
            "interface_sha256": sha256_bytes(interface_bytes),
            "archive_members": ["README.txt", interface_name],
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "handoff_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
