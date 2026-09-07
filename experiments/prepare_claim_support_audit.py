#!/usr/bin/env python3
"""Prepare a blinded, stratified, two-annotator RAES audit bundle.

The sampling unit is a model output.  The default design samples 25 common
tasks across four systems (100 outputs) and extracts at most five cited
claim--source pairs per output (at most 500 pairs).  System identities are
kept only in the master files; annotator sheets use opaque IDs and independent
row orders.

This script prepares annotation material.  It does not create human labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


URL_RE = re.compile(r"https?://[^\s\)\]\}\>\"']+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)")
SOURCE_RE = re.compile(r"(?:source|sources)\s*:\s*(.*)", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
GROUP_MAP = {
    "answerable": "answerable",
    "answerable_with_uncertainty": "uncertainty",
    "conflicting_no_resolution": "conflict",
    "not_enough_public_evidence": "insufficient-evidence",
}
GROUP_ORDER = ("answerable", "uncertainty", "conflict", "insufficient-evidence")
HUMAN_CSV_FIELDS = {
    "support_label",
    "source_policy_compliant",
    "conflict_disclosure_correct",
    "uncertainty_present",
    "uncertainty_appropriate",
    "stop_decision",
    "rationale",
}
HUMAN_CSV_FILES = (
    "annotator_a_claim_labels.csv",
    "annotator_b_claim_labels.csv",
    "annotator_a_output_labels.csv",
    "annotator_b_output_labels.csv",
    "adjudicated_claim_labels.csv",
    "adjudicated_output_labels.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system-results",
        nargs="+",
        required=True,
        metavar="LABEL=RESULTS_JSONL",
        help="One result file per system. IDs must overlap across all systems.",
    )
    parser.add_argument(
        "--system-scores",
        nargs="*",
        default=[],
        metavar="LABEL=SCORES_JSONL",
        help="Optional automatic score files; E is copied into the master manifest.",
    )
    parser.add_argument("--gold", required=True, help="Gold JSONL with answerability and policy fields.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", type=int, default=25)
    parser.add_argument("--pairs-per-output", type=int, default=5)
    parser.add_argument("--max-claim-pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260519)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def parse_labeled_paths(values: Iterable[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got: {value}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or label in parsed:
            raise ValueError(f"Missing or duplicate system label: {label!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        parsed[label] = path
    return parsed


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("id") or "")


def normalize_url(url: str) -> str:
    return url.strip().rstrip(".,;:").split("#", 1)[0].rstrip("/").lower()


def visible_answer(text: str) -> str:
    """Remove model reasoning blocks before auditing user-visible claims."""
    cleaned = THINK_BLOCK_RE.sub("", str(text or ""))
    # Be conservative with a dangling closing tag emitted by some local
    # Qwen checkpoints: content after it is the user-facing answer.
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>\s*", cleaned, flags=re.IGNORECASE)[-1]
    cleaned = re.sub(r"</?think\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def citation_lookup(row: dict[str, Any]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for citation in row.get("citations") or []:
        url = str(citation.get("url") or "")
        if not url:
            continue
        lookup[normalize_url(url)] = {
            "title": str(citation.get("title") or ""),
            "snippet": str(citation.get("snippet") or ""),
            "source_type": str(citation.get("source_type") or ""),
        }
    return lookup


def clean_claim(text: str) -> str:
    text = MARKDOWN_LINK_RE.sub(lambda match: match.group(1), text)
    text = URL_RE.sub("", text)
    text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text)
    text = re.sub(r"[*_`#]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -:;.")


def is_source_only(text: str) -> bool:
    cleaned = clean_claim(text).lower()
    return not cleaned or cleaned.startswith(("source", "sources", "reference", "references"))


def citation_claim_context(line: str, url_start: int, url_end: int) -> str:
    """Return the sentence containing one citation, without the URL itself.

    This keeps paragraph-style cited answers auditable instead of assuming that
    every claim is a bullet.  Markdown link text is preserved as prose, while a
    naked URL is replaced by a marker before sentence segmentation.
    """
    marker = " __EVISUFF_CITATION__ "
    masked = ""
    for match in MARKDOWN_LINK_RE.finditer(line):
        if match.start(2) <= url_start < match.end(2):
            masked = (
                line[: match.start()]
                + match.group(1)
                + marker
                + line[match.end() :]
            )
            break
    if not masked:
        masked = line[:url_start] + marker + line[url_end:]
    sentences = re.split(r"(?<=[.!?])\s+", masked)
    containing = next((sentence for sentence in sentences if marker.strip() in sentence), masked)
    return clean_claim(containing.replace(marker.strip(), ""))


def pair_row(claim: str, url: str, lookup: dict[str, dict[str, str]]) -> dict[str, str]:
    norm_url = normalize_url(url)
    source = lookup.get(norm_url, {})
    return {
        "claim": clean_claim(claim),
        "citation_url": norm_url,
        "citation_title": source.get("title", ""),
        "citation_snippet": source.get("snippet", ""),
        "source_type": source.get("source_type", ""),
    }


def extract_claim_pairs(row: dict[str, Any]) -> list[dict[str, str]]:
    """Extract cited claims conservatively from common RAES answer formats."""
    answer = visible_answer(str(row.get("final_answer") or ""))
    lookup = citation_lookup(row)
    pairs: list[dict[str, str]] = []
    pending_claim = ""
    in_bibliography = False
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"#{1,6}.*", line):
            heading = re.sub(r"^#{1,6}\s*", "", line).strip().casefold()
            in_bibliography = heading.startswith(("source", "reference", "bibliography"))
            continue
        plain_heading = clean_claim(line).rstrip(":").casefold()
        if plain_heading in {"sources", "references", "bibliography"}:
            in_bibliography = True
            continue
        if in_bibliography:
            continue
        url_matches = list(URL_RE.finditer(line))
        urls = [match.group(0) for match in url_matches]
        source_match = SOURCE_RE.search(line)
        if source_match and urls:
            claim = pending_claim
            if claim:
                pairs.extend(pair_row(claim, url, lookup) for url in urls)
            pending_claim = ""
            continue
        if urls:
            for match in url_matches:
                claim = citation_claim_context(line, match.start(), match.end())
                if not is_source_only(claim) and claim:
                    pairs.append(pair_row(claim, match.group(0), lookup))
                elif pending_claim:
                    pairs.append(pair_row(pending_claim, match.group(0), lookup))
            pending_claim = ""
            continue
        candidate = clean_claim(line)
        if candidate and len(candidate) < 1000:
            # Retain ordinary prose as well as bullets so a following
            # ``Source: URL`` line can be paired with its claim.
            if pending_claim and line[:1].islower():
                pending_claim = f"{pending_claim} {candidate}".strip()
            else:
                pending_claim = candidate

    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        key = (pair["claim"].casefold(), pair["citation_url"])
        if pair["claim"] and pair["citation_url"] and key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique


def gold_policy(row: dict[str, Any]) -> dict[str, Any]:
    gold_answer = row.get("gold_answer") or {}
    must_include = gold_answer.get("must_include") or {}
    gold_sources = []
    for source in row.get("gold_sources") or []:
        gold_sources.append({
            "title": str(source.get("title") or ""),
            "url": str(source.get("url") or ""),
            "source_type": str(source.get("source_type") or ""),
            "authority_level": str(source.get("authority_level") or ""),
            "role": str(source.get("role") or ""),
        })
    conflicts = []
    for conflict in row.get("known_conflicts") or []:
        conflicts.append({
            "claim": str(conflict.get("claim") or conflict.get("description") or ""),
            "positions": conflict.get("positions") or conflict.get("sides") or [],
            "resolution": str(
                conflict.get("resolution")
                or conflict.get("resolution_status")
                or conflict.get("handling")
                or ""
            ),
        })
    return {
        "gold_answerability": str(row.get("answerability") or gold_answer.get("answerability") or ""),
        "uncertainty_requirement": str(gold_answer.get("uncertainty_requirement") or ""),
        "boundary_conditions": must_include.get("boundary_conditions") or [],
        "required_facets": row.get("required_facets") or {},
        "evidence_points": must_include.get("evidence_points") or [],
        "gold_final_answer": str(gold_answer.get("final_answer") or ""),
        "source_policy": row.get("source_policy") or {},
        "minimum_evidence_policy": row.get("minimum_evidence_policy") or {},
        "gold_source_inventory": gold_sources,
        "known_conflicts": conflicts,
        "stop_condition": row.get("stop_condition") or {},
    }


def score_index(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in load_jsonl(path):
        sid = row_id(row)
        for key in ("evidence_support", "evidence", "E"):
            try:
                if sid and row.get(key) is not None:
                    out[sid] = float(row[key])
                    break
            except (TypeError, ValueError):
                pass
    return out


def compact_citation_inventory(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return retrieved sources with final-answer citation status recomputed.

    Some historical RAES runs leave the pipeline-level ``cited`` flag false
    even when the URL is visibly cited in ``final_answer``.  Human annotators
    must therefore not rely on that implementation flag.  We derive
    ``appears_in_final_answer`` from the answer text and retain the raw flag
    only for traceability.
    """
    answer_url_map = {
        normalize_url(url): url
        for url in URL_RE.findall(visible_answer(str(row.get("final_answer") or "")))
    }
    inventory = []
    seen: set[str] = set()
    for citation in row.get("citations") or []:
        url = str(citation.get("url") or "")
        if not url:
            continue
        normalized_url = normalize_url(url)
        if normalized_url in seen:
            continue
        seen.add(normalized_url)
        inventory.append({
            "url": url,
            "title": str(citation.get("title") or ""),
            "source_type": str(citation.get("source_type") or ""),
            "appears_in_final_answer": normalized_url in answer_url_map,
            "pipeline_cited_flag": bool(citation.get("cited", False)),
        })
    # Preserve citations that appear in the answer even if the runtime failed
    # to copy them into its citation inventory.
    for normalized_url, original_url in answer_url_map.items():
        if normalized_url not in seen:
            inventory.append({
                "url": original_url,
                "title": "",
                "source_type": "",
                "appears_in_final_answer": True,
                "pipeline_cited_flag": False,
            })
    return inventory


def compact_gate_trace(row: dict[str, Any]) -> list[dict[str, Any]]:
    trace = []
    for decision in row.get("gate_decisions") or []:
        trace.append({
            "decision": str(decision.get("decision") or ""),
            "passed": decision.get("passed"),
            "failed_checks": decision.get("failed_checks") or [],
            "missing_facets": decision.get("missing_facets") or [],
            "unsupported_claim_count": len(decision.get("unsupported_claims") or []),
            "forced_by_gate_budget": bool(decision.get("forced_by_gate_budget", False)),
        })
    return trace


def compact_action_trace(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Load a compact, model-blind tool trace for human stopping judgments."""
    raw_path = str(row.get("trajectory_path") or "").strip()
    if not raw_path:
        return []
    path = Path(raw_path)
    if not path.is_file():
        return []
    actions: list[dict[str, Any]] = []
    by_use_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            event_type = str(event.get("event_type") or "")
            data = event.get("data") or {}
            if event_type == "tool_use":
                action = {
                    "step": event.get("step"),
                    "tool": str(data.get("tool_name") or ""),
                    "input": data.get("tool_input") or {},
                    "is_error": None,
                    "result_chars": None,
                    "result_preview": "",
                }
                actions.append(action)
                use_id = str(data.get("tool_use_id") or "")
                if use_id:
                    by_use_id[use_id] = action
            elif event_type == "tool_result":
                use_id = str(data.get("tool_use_id") or "")
                action = by_use_id.get(use_id)
                if action is None:
                    continue
                result = re.sub(r"\s+", " ", str(data.get("result") or "")).strip()
                action["is_error"] = bool(data.get("is_error", False))
                action["result_chars"] = int(data.get("result_chars") or len(result))
                action["result_preview"] = result[:320]
    return actions


def choose_tasks(
    common_ids: set[str], gold: dict[str, dict[str, Any]], target: int, rng: random.Random
) -> tuple[list[str], dict[str, int]]:
    by_group: dict[str, list[str]] = defaultdict(list)
    for sid in sorted(common_ids):
        answerability = str(gold[sid].get("answerability") or "")
        group = GROUP_MAP.get(answerability)
        if group:
            by_group[group].append(sid)
    for ids in by_group.values():
        rng.shuffle(ids)

    target = min(target, sum(len(by_group[group]) for group in GROUP_ORDER))
    base = target // len(GROUP_ORDER)
    quota = {group: min(base, len(by_group[group])) for group in GROUP_ORDER}
    remaining = target - sum(quota.values())
    while remaining:
        progressed = False
        for group in GROUP_ORDER:
            if quota[group] < len(by_group[group]):
                quota[group] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break
    selected = [sid for group in GROUP_ORDER for sid in by_group[group][: quota[group]]]
    rng.shuffle(selected)
    return selected, quota


def opaque_id(prefix: str, raw: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{raw}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def existing_human_work(output_dir: Path) -> list[str]:
    """Return protected artifacts that contain annotation/provenance work."""
    protected: list[str] = []
    if (output_dir / ".human_audit_complete").exists():
        protected.append(".human_audit_complete")
    for filename in HUMAN_CSV_FILES:
        path = output_dir / filename
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            protected.append(f"{filename} (unreadable)")
            continue
        if any(
            str(row.get(field) or "").strip()
            for row in rows
            for field in HUMAN_CSV_FIELDS
        ):
            protected.append(filename)

    provenance_path = output_dir / "HUMAN_ANNOTATION_PROVENANCE.json"
    if provenance_path.exists():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            annotators = provenance.get("annotators") or []
            has_provenance = any(
                str(row.get("anonymous_id") or "").strip()
                or row.get("is_human") is not None
                or row.get("independent_annotation") is not None
                or str(row.get("completion_date") or "").strip()
                for row in annotators
            ) or bool(
                str(provenance.get("adjudicator_anonymous_id") or "").strip()
                or str(provenance.get("adjudication_completion_date") or "").strip()
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            has_provenance = True
        if has_provenance:
            protected.append("HUMAN_ANNOTATION_PROVENANCE.json")
    return protected


def main() -> None:
    args = parse_args()
    if args.tasks <= 0 or args.pairs_per_output <= 0 or args.max_claim_pairs <= 0:
        raise SystemExit("--tasks, --pairs-per-output, and --max-claim-pairs must be positive")
    output_dir = Path(args.output_dir)
    protected = existing_human_work(output_dir)
    if protected:
        raise SystemExit(
            "Refusing to overwrite existing human annotation work: "
            + ", ".join(protected)
            + ". Archive or move the audit directory before regenerating it."
        )
    rng = random.Random(args.seed)
    result_paths = parse_labeled_paths(args.system_results)
    score_paths = parse_labeled_paths(args.system_scores)
    unknown_scores = set(score_paths) - set(result_paths)
    if unknown_scores:
        raise SystemExit(f"Scores supplied for unknown systems: {sorted(unknown_scores)}")

    gold_rows = load_jsonl(Path(args.gold))
    gold = {row_id(row): row for row in gold_rows if row_id(row)}
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for system, path in result_paths.items():
        indexed = {row_id(row): row for row in load_jsonl(path) if row_id(row)}
        results[system] = indexed
    common_ids = set(gold)
    for indexed in results.values():
        common_ids &= set(indexed)
    selected_ids, quotas = choose_tasks(common_ids, gold, args.tasks, rng)
    if len(selected_ids) < args.tasks:
        raise SystemExit(f"Only {len(selected_ids)} eligible common tasks; requested {args.tasks}")

    scores = {system: score_index(path) for system, path in score_paths.items()}
    output_dir.mkdir(parents=True, exist_ok=True)

    output_master: list[dict[str, Any]] = []
    pair_master: list[dict[str, Any]] = []
    for system, indexed in results.items():
        for sid in selected_ids:
            row = indexed[sid]
            raw_output_id = f"{system}::{sid}"
            blind_output_id = opaque_id("out", raw_output_id, args.seed)
            policy = gold_policy(gold[sid])
            group = GROUP_MAP[policy["gold_answerability"]]
            citation_inventory = compact_citation_inventory(row)
            action_trace = compact_action_trace(row)
            output_master.append({
                "output_id": raw_output_id,
                "blind_output_id": blind_output_id,
                "system": system,
                "sample_id": sid,
                "group": group,
                "question": str(row.get("question") or gold[sid].get("question") or ""),
                "final_answer": visible_answer(str(row.get("final_answer") or "")),
                "raw_final_answer": str(row.get("final_answer") or ""),
                "search_calls": int((row.get("tool_stats") or {}).get("search_calls") or 0),
                "fetch_calls": int((row.get("tool_stats") or {}).get("fetch_calls") or 0),
                # The runtime counters cap executed search/fetch operations at
                # the configured budget.  Human stopping judgments must also
                # see rejected attempts after the cap and calls to disabled or
                # auxiliary tools, so total attempts are defined by the raw
                # tool-use trace rather than the capped execution counter.
                "total_tool_calls": len(action_trace),
                "retrieved_source_count": len(citation_inventory),
                "final_citation_count": sum(
                    bool(item["appears_in_final_answer"])
                    for item in citation_inventory
                ),
                "citation_inventory": citation_inventory,
                "action_trace": action_trace,
                "gate_trace": compact_gate_trace(row),
                "automatic_E": scores.get(system, {}).get(sid),
                **policy,
            })
            pairs = extract_claim_pairs(row)
            random.Random(f"{args.seed}:{raw_output_id}").shuffle(pairs)
            pairs = pairs[: args.pairs_per_output]
            for pair_no, pair in enumerate(pairs, 1):
                raw_pair_id = f"{raw_output_id}::c{pair_no}"
                pair_master.append({
                    "pair_id": raw_pair_id,
                    "blind_pair_id": opaque_id("pair", raw_pair_id, args.seed),
                    "output_id": raw_output_id,
                    "blind_output_id": blind_output_id,
                    "system": system,
                    "sample_id": sid,
                    "group": group,
                    "question": str(row.get("question") or gold[sid].get("question") or ""),
                    **pair,
                })

    if len(pair_master) > args.max_claim_pairs:
        # Preserve output coverage before allocating remaining pair slots.
        first: dict[str, dict[str, Any]] = {}
        rest: list[dict[str, Any]] = []
        for pair in pair_master:
            if pair["output_id"] not in first:
                first[pair["output_id"]] = pair
            else:
                rest.append(pair)
        rng.shuffle(rest)
        pair_master = list(first.values()) + rest[: max(0, args.max_claim_pairs - len(first))]

    write_jsonl(output_dir / "outputs_master.jsonl", output_master)
    write_jsonl(output_dir / "claim_pairs_master.jsonl", pair_master)

    claim_fields = [
        "blind_pair_id", "blind_output_id", "question", "claim",
        "citation_url", "citation_title", "citation_snippet", "source_type",
        "support_label", "rationale",
    ]
    output_fields = [
        "blind_output_id", "group", "question", "final_answer", "gold_final_answer",
        "uncertainty_requirement", "boundary_conditions", "required_facets",
        "evidence_points", "source_policy", "minimum_evidence_policy",
        "gold_source_inventory", "known_conflicts", "stop_condition",
        "search_calls", "fetch_calls", "total_tool_calls",
        "retrieved_source_count", "final_citation_count",
        "citation_inventory", "action_trace", "gate_trace",
        "source_policy_compliant", "conflict_disclosure_correct",
        "uncertainty_present", "uncertainty_appropriate", "stop_decision",
        "rationale",
    ]
    claim_rows = [{**row, "support_label": "", "rationale": ""} for row in pair_master]
    output_rows = [{**row, "source_policy_compliant": "", "conflict_disclosure_correct": "",
                    "uncertainty_present": "", "uncertainty_appropriate": "",
                    "stop_decision": "", "rationale": ""} for row in output_master]
    for annotator, offset in (("a", 101), ("b", 202)):
        claim_copy = claim_rows.copy()
        output_copy = output_rows.copy()
        random.Random(args.seed + offset).shuffle(claim_copy)
        random.Random(args.seed + offset + 1).shuffle(output_copy)
        write_csv(output_dir / f"annotator_{annotator}_claim_labels.csv", claim_copy, claim_fields)
        write_csv(output_dir / f"annotator_{annotator}_output_labels.csv", output_copy, output_fields)
    # The adjudicator fills these after reviewing disagreements. Keeping full
    # templates makes ID validation deterministic and preserves original labels.
    write_csv(output_dir / "adjudicated_claim_labels.csv", claim_rows, claim_fields)
    write_csv(output_dir / "adjudicated_output_labels.csv", output_rows, output_fields)

    group_counts = Counter(row["group"] for row in output_master)
    pair_group_counts = Counter(row["group"] for row in pair_master)
    pair_system_counts = Counter(row["system"] for row in pair_master)
    paired_outputs = {row["output_id"] for row in pair_master}
    outputs_without_pairs = Counter(
        row["system"] for row in output_master if row["output_id"] not in paired_outputs
    )
    manifest = {
        "design": "common-task, system-blinded, two-annotator audit",
        "seed": args.seed,
        "systems": list(result_paths),
        "common_task_pool": len(common_ids),
        "selected_tasks": len(selected_ids),
        "task_quota_per_group": quotas,
        "outputs": len(output_master),
        "outputs_per_group": dict(group_counts),
        "claim_source_pairs": len(pair_master),
        "outputs_with_claim_pairs": len(paired_outputs),
        "outputs_without_claim_pairs": len(output_master) - len(paired_outputs),
        "outputs_without_claim_pairs_per_system": dict(outputs_without_pairs),
        "claim_pairs_per_group": dict(pair_group_counts),
        "claim_pairs_per_system": dict(pair_system_counts),
        "pairs_per_output_cap": args.pairs_per_output,
        "claim_pair_cap": args.max_claim_pairs,
        "human_labels_present": False,
        "note": "Blank sheets require two independent human annotators; no labels are synthesized.",
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    instructions = """# Human audit instructions

Annotators must work independently and must not attempt to infer model identity.
Claim sheets hide both model identity and the gold answerability condition;
output sheets expose the gold requirement only because it is necessary to
judge uncertainty and stopping. The independent statistical unit is the output
(`blind_output_id`), not each claim--source pair.

## Claim labels

- Open `citation_url` and judge the claim against the source itself whenever
  the page is accessible; `citation_snippet` is navigation context, not a
  substitute for the source. Check the relevant passage and qualifications,
  not merely title or topical similarity.
- `supported`: the cited source directly entails the material claim.
- `partially_supported`: the source supports only part of the claim or supports
  it with a material qualification omitted by the answer.
- `unsupported`: the source does not support the claim, contradicts it, or is
  merely topically related.
- `unclear`: the source cannot be accessed or its available content is
  genuinely insufficient to judge. Do not convert inaccessible evidence into
  `unsupported` unless the accessible source content actually fails to support
  the claim.

## Output labels

- `source_policy_compliant`: `yes`/`no` using the supplied evidence and boundary
  requirements, especially `source_policy`, `minimum_evidence_policy`, and
  `gold_source_inventory`. Judge sources attached in the final answer. The gold
  inventory gives validated examples; do not require an exact URL match when the
  policy permits an equivalent authoritative source. Conversely, topical or
  secondary sources do not satisfy a policy that explicitly requires a named or
  primary source. Inventory entries
  with `appears_in_final_answer=false` are retrieved context, not support for
  final claims. `pipeline_cited_flag` is retained only for provenance and must
  not override the recomputed final-answer status.
- `conflict_disclosure_correct`: `yes`/`no`/`not_applicable`, using
  `known_conflicts` and the gold boundary. Mark `yes` only if the answer states
  the material disagreement without falsely resolving it; mark `no` if it
  suppresses, distorts, or over-resolves that disagreement. Use `not_applicable`
  only when the task has no disclosure-relevant conflict.
- `uncertainty_present`: `yes` only when the visible final answer explicitly
  communicates an evidence limitation, unresolved disagreement, range, caveat,
  or inability to determine the requested fact. Generic politeness or vague
  hedging alone is `no`. This label is descriptive, regardless of correctness.
- `uncertainty_appropriate`: `yes` when the amount and reason for uncertainty
  match the supplied gold boundary. For answerable tasks, unwarranted material
  doubt is inappropriate, although a precise non-material caveat is allowed.
  For uncertainty, conflict, or insufficient-evidence tasks, the answer must
  identify the relevant limitation and must not assert beyond it. Otherwise use
  `no`.
- `stop_decision` is mutually exclusive and uses this precedence:
  1. `premature` if the final state still violates a required facet, source
     minimum, conflict/uncertainty boundary, or explicit stop condition and the
     answer nevertheless finalizes. Repeated but unproductive searching does
     not erase a premature finalization.
  2. Otherwise, `unnecessary_search` if the requirements had already been met
     and the trace then contains clearly redundant or off-boundary search that
     was not needed to verify or stabilize the answer.
  3. Otherwise, `appropriate`. Stopping with an explicit evidence boundary is
     appropriate when the task is genuinely unanswerable or conflicting and
     reasonable acquisition has satisfied the supplied stop condition.

Use the search/fetch counts, citation inventory, compact action trace, and
gate trace when
judging stopping behavior, together with `required_facets`,
`minimum_evidence_policy`, and `stop_condition`. Do not penalize a system
merely for a larger count; judge whether the additional search was necessary
for the task boundary. Apply the same rules to every blinded output; do not
infer system identity from style, latency, or tool count.

Write a short rationale for non-obvious or negative judgments. After both
annotators finish, adjudicate disagreements in separate files; never overwrite
the original sheets.
"""
    (output_dir / "ANNOTATION_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    provenance_path = output_dir / "HUMAN_ANNOTATION_PROVENANCE.json"
    if not provenance_path.exists():
        provenance = {
            "annotation_protocol": "independent double human annotation followed by adjudication",
            "annotators": [
                {
                    "sheet": "a",
                    "anonymous_id": "",
                    "is_human": None,
                    "independent_annotation": None,
                    "completion_date": "",
                    "conflict_of_interest": "",
                },
                {
                    "sheet": "b",
                    "anonymous_id": "",
                    "is_human": None,
                    "independent_annotation": None,
                    "completion_date": "",
                    "conflict_of_interest": "",
                },
            ],
            "adjudicator_anonymous_id": "",
            "adjudication_completion_date": "",
            "declaration": "AI or LLM pre-annotations, if any, are not counted as human labels.",
        }
        provenance_path.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
