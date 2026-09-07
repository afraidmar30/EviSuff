#!/usr/bin/env python3
"""Verify the final ICASSP paper, human audit, and five-page boundary."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from experiments.verify_sufficiency_audit_artifacts import validate_core_latex_table


REFERENCE_HEADING = re.compile(r"^\s*(?:\d+\.\s*)?REFERENCES\b", re.IGNORECASE)
BAD_LOG_PATTERNS = {
    "overfull_hbox": re.compile(r"Overfull \\hbox", re.IGNORECASE),
    "overfull_vbox": re.compile(r"Overfull \\vbox", re.IGNORECASE),
    "float_too_large": re.compile(r"Float too large", re.IGNORECASE),
    "undefined_reference": re.compile(r"(?:Reference .* undefined|undefined references)", re.IGNORECASE),
    "undefined_citation": re.compile(r"(?:Citation .* undefined|undefined citations)", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--paper-tex", default="ICASSP2026_Paper_Templates/EviSuff_ICASSP2026.tex"
    )
    parser.add_argument(
        "--paper-pdf", default="ICASSP2026_Paper_Templates/EviSuff_ICASSP2026.pdf"
    )
    parser.add_argument(
        "--paper-log", default="ICASSP2026_Paper_Templates/EviSuff_ICASSP2026.log"
    )
    parser.add_argument(
        "--audit-dir", default="outputs/audits/sufficiency_audit25_four_systems"
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def collect_local_tex_sources(path: Path, seen: set[Path] | None = None) -> str:
    """Read a TeX source and resolvable local inputs, without following packages."""
    path = path.resolve()
    seen = seen or set()
    if path in seen or not path.is_file():
        return ""
    seen.add(path)
    text = path.read_text(encoding="utf-8")

    def expand(match: re.Match[str]) -> str:
        raw = match.group(1)
        child = Path(raw)
        if not child.suffix:
            child = child.with_suffix(".tex")
        if not child.is_absolute():
            child = path.parent / child
        if child.is_file():
            return match.group(0) + "\n" + collect_local_tex_sources(child, seen)
        return match.group(0)

    return re.sub(r"\\(?:input|include)\s*\{([^}]+)\}", expand, text)


def references_only_on_last_page(page4: str, page5: str) -> tuple[bool, dict[str, Any]]:
    """Require the references heading to start page 5 and be absent from page 4."""
    page4_lines = [line.strip() for line in page4.replace("\f", "").splitlines()]
    page5_lines = [line.strip() for line in page5.replace("\f", "").splitlines()]
    page4_headings = [line for line in page4_lines if REFERENCE_HEADING.match(line)]
    heading_indices = [index for index, line in enumerate(page5_lines) if REFERENCE_HEADING.match(line)]
    prefix = page5_lines[:heading_indices[0]] if heading_indices else page5_lines
    substantive_prefix = [line for line in prefix if line and not re.fullmatch(r"\d+", line)]
    has_entries = any(re.match(r"^\[?1\]?\b", line) for line in page5_lines)
    passed = (
        not page4_headings
        and len(heading_indices) == 1
        and not substantive_prefix
        and has_entries
    )
    return passed, {
        "page4_reference_headings": page4_headings,
        "page5_reference_heading_count": len(heading_indices),
        "page5_text_before_heading": substantive_prefix[:10],
        "page5_has_reference_1": has_entries,
    }


def latex_log_issues(text: str) -> list[str]:
    return [name for name, pattern in BAD_LOG_PATTERNS.items() if pattern.search(text)]


def validate_paper_audit_table(text: str, report: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Match the compact manuscript table to authoritative overall estimates."""
    system_names = {
        "Base": "base",
        "Answer-SFT": "answer-sft",
        "No-gate": "no-gate",
        "Full EviSuff": "full-evisuff",
    }
    metrics = (
        "claim_support_rate",
        "source_policy_compliance",
        "uncertainty_calibration_accuracy",
        "stop_decision_accuracy",
        "premature_finalization_rate",
    )
    normalized = text.replace("$-$", "-").replace(r"\textbf{", "")
    normalized = normalized.replace("{", "").replace("}", "")
    rows = {}
    for line in normalized.splitlines():
        if "&" not in line:
            continue
        cells = [cell.strip() for cell in line.replace(r"\\", "").split("&")]
        if cells:
            rows[cells[0]] = cells[1:]

    mismatches = []
    for display, system in system_names.items():
        observed = rows.get(display)
        expected = [
            f"{100 * float(report['direct'][system]['overall']['metrics'][metric]['estimate']):.1f}"
            for metric in metrics
        ]
        if observed != expected:
            mismatches.append({"row": display, "observed": observed, "expected": expected})

    contrast = report["paired_direct_contrasts"]["no-gate"]["overall"]["metrics"]
    expected_delta = []
    for metric in metrics:
        item = contrast[metric]
        estimate = 100 * float(item["estimate"])
        low = 100 * float(item["ci_low"])
        high = 100 * float(item["ci_high"])
        expected_delta.append(f"{estimate:+.1f} [{low:.1f},{high:.1f}]")
    observed_delta = rows.get("Full - No-gate")
    if observed_delta != expected_delta:
        mismatches.append({
            "row": "Full - No-gate", "observed": observed_delta, "expected": expected_delta,
        })

    required = all(token in text for token in (
        r"\label{tab:human-audit}", "CSR$\\uparrow$", "SPC$\\uparrow$",
        "U-Cal$\\uparrow$", "Stop$\\uparrow$", "Prem.$\\downarrow$",
    ))
    return required and not mismatches, {
        "required_tokens_present": required,
        "rows_checked": len(system_names) + 1,
        "mismatches": mismatches,
    }


def command_output(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return completed.stdout


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    paper_tex = (root / args.paper_tex).resolve()
    paper_pdf = (root / args.paper_pdf).resolve()
    paper_log = (root / args.paper_log).resolve()
    audit_dir = (root / args.audit_dir).resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    completion_marker = audit_dir / ".human_audit_complete"
    check("human_audit_completion_marker", completion_marker.is_file(), str(completion_marker))

    verification_path = audit_dir / "final_verification.json"
    verification = (
        json.loads(verification_path.read_text(encoding="utf-8"))
        if verification_path.is_file() else {}
    )
    check(
        "strict_human_audit_verification_passed",
        verification.get("status") == "passed"
        and verification.get("require_human_labels") is True,
        {"path": str(verification_path), "status": verification.get("status")},
    )

    report_path = audit_dir / "report.json"
    table_path = audit_dir / "direct_table.tex"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    table_text = table_path.read_text(encoding="utf-8") if table_path.is_file() else ""
    table_ok, table_detail = validate_core_latex_table(table_text, report)
    check("authoritative_core_table_valid", table_ok, table_detail)

    combined_source = collect_local_tex_sources(paper_tex)
    compact_table_ok, compact_table_detail = validate_paper_audit_table(combined_source, report)
    check("paper_compact_audit_table_matches_report", compact_table_ok, compact_table_detail)
    check(
        "paper_source_has_bibliography_after_audit_table",
        combined_source.find(r"\label{tab:human-audit}") >= 0
        and combined_source.rfind(r"\bibliography{")
        > combined_source.find(r"\label{tab:human-audit}"),
        "human-audit table precedes bibliography",
    )

    pdf_exists = paper_pdf.is_file() and paper_pdf.stat().st_size > 0
    check("paper_pdf_exists", pdf_exists, str(paper_pdf))
    pages = None
    page4 = page5 = ""
    if pdf_exists:
        info = command_output(["pdfinfo", str(paper_pdf)])
        match = re.search(r"^Pages:\s*(\d+)\s*$", info, re.MULTILINE)
        pages = int(match.group(1)) if match else None
        if pages and pages >= 5:
            page4 = command_output(["pdftotext", "-f", "4", "-l", "4", "-layout", str(paper_pdf), "-"])
            page5 = command_output(["pdftotext", "-f", "5", "-l", "5", "-layout", str(paper_pdf), "-"])
    check("paper_is_exactly_five_pages", pages == 5, pages)
    boundary_ok, boundary_detail = references_only_on_last_page(page4, page5)
    check("body_pages_1_to_4_references_only_page_5", boundary_ok, boundary_detail)
    check(
        "rendered_pdf_contains_core_table",
        "CSR" in page4 and "Full EviSuff" in page4,
        {"CSR_on_page4": "CSR" in page4, "Full_EviSuff_on_page4": "Full EviSuff" in page4},
    )

    log_text = paper_log.read_text(encoding="utf-8", errors="replace") if paper_log.is_file() else ""
    issues = latex_log_issues(log_text)
    check("latex_log_clean", paper_log.is_file() and not issues, {"path": str(paper_log), "issues": issues})
    newest_source = max([
        paper_tex.stat().st_mtime if paper_tex.is_file() else 0,
        report_path.stat().st_mtime if report_path.is_file() else 0,
    ])
    check(
        "pdf_not_older_than_paper_or_core_table",
        pdf_exists and paper_pdf.stat().st_mtime >= newest_source,
        {
            "pdf_mtime": paper_pdf.stat().st_mtime if pdf_exists else None,
            "newest_required_source_mtime": newest_source,
        },
    )

    passed = all(item["passed"] for item in checks)
    payload = {
        "status": "passed" if passed else "failed",
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# ICASSP submission-readiness verification", "",
        f"Status: **{payload['status']}**", "",
        "| Check | Pass | Detail |", "|---|---:|---|",
    ]
    for item in checks:
        detail = json.dumps(item["detail"], ensure_ascii=False).replace("|", r"\|")
        lines.append(f"| {item['name']} | {'yes' if item['passed'] else 'NO'} | `{detail}` |")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "checks_passed", "checks_total")}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
