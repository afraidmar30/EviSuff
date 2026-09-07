"""Conflict and uncertainty checks for RAES-Gate."""

from __future__ import annotations

import json

from src.raes_eval.gate_prompts import CONFLICT_UNCERTAINTY_PROMPT
from src.raes_eval.gate_utils import json_side_query, source_summaries
from src.raes_eval.schemas import DraftAnswer, GateDecision, RAESTask, RetrievedSource


CONFLICT_SCHEMA = {
    "type": "object",
    "properties": {
        "has_conflict": {"type": "boolean"},
        "conflict_summary": {"type": "string"},
        "uncertainty_needed": {"type": "boolean"},
        "uncertainty_missing": {"type": "boolean"},
        "should_abstain": {"type": "boolean"},
        "required_revision": {"type": "string"},
        "decision": {"type": "string"},
    },
    "required": [
        "has_conflict",
        "conflict_summary",
        "uncertainty_needed",
        "uncertainty_missing",
        "should_abstain",
        "required_revision",
        "decision",
    ],
    "additionalProperties": False,
}

UNCERTAINTY_TASK_TYPES = {
    "documentation_change_tracking",
    "current_status_verification",
    "live_fact_verification",
    "leaderboard_verification",
    "adversarial_conflict_resolution",
}


async def detect_conflicts_and_uncertainty(
    task: RAESTask,
    draft: DraftAnswer,
    sources: list[RetrievedSource],
) -> GateDecision | None:
    prompt = json.dumps(
        {
            "public_task": {
                "id": task.id,
                "category": task.category,
                "domain": task.domain,
                "task_type": task.task_type,
                "difficulty": task.difficulty,
                "question": task.question,
            },
            "draft_answer": draft.text[:7000],
            "sources": source_summaries(sources, max_text_chars=800),
        },
        ensure_ascii=False,
    )
    parsed = await json_side_query(
        prompt=prompt,
        system=CONFLICT_UNCERTAINTY_PROMPT,
        output_schema=CONFLICT_SCHEMA,
        max_tokens=900,
    )
    if not isinstance(parsed, dict):
        if _needs_uncertainty(task) and not _has_uncertainty_language(draft.text):
            return GateDecision(
                decision="add_uncertainty",
                passed=False,
                failed_checks=["uncertainty_verifier_unavailable", "missing_uncertainty_boundary"],
                reason="Uncertainty verifier failed; conservative fallback requires an uncertainty or time/version boundary.",
                required_next_actions=["Add uncertainty, access-date, version, or evidence boundary language."],
                revised_answer_hint="Revise the answer to state what is supported, what remains uncertain, and the time/version boundary.",
                confidence=0.45,
                layer_results=[{"layer": "conflict_uncertainty", "fallback": True}],
            )
        return None

    has_conflict = bool(parsed.get("has_conflict"))
    uncertainty_needed = bool(parsed.get("uncertainty_needed"))
    uncertainty_missing = bool(parsed.get("uncertainty_missing"))
    should_abstain = bool(parsed.get("should_abstain"))
    required_revision = str(parsed.get("required_revision", "")).strip()
    decision = str(parsed.get("decision", "") or "allow_final")

    if should_abstain:
        return GateDecision(
            decision="abstain",
            passed=False,
            failed_checks=["should_abstain"],
            reason=required_revision or "Evidence is insufficient for a reliable answer.",
            unresolved_conflicts=[str(parsed.get("conflict_summary", ""))] if has_conflict else [],
            required_next_actions=["Abstain or answer only with clearly stated uncertainty."],
            revised_answer_hint=required_revision or "Say the available evidence is insufficient for a reliable conclusion.",
            confidence=0.75,
            layer_results=[{"layer": "conflict_uncertainty", **parsed}],
        )

    failed = []
    if has_conflict and decision in {"continue_search", "revise_answer"}:
        failed.append("unresolved_conflict")
    if uncertainty_needed and uncertainty_missing:
        failed.append("missing_uncertainty_statement")

    if not failed:
        return None

    return GateDecision(
        decision="add_uncertainty" if "missing_uncertainty_statement" in failed else "revise_answer",
        passed=False,
        failed_checks=failed,
        reason=required_revision or "Conflict / Uncertainty Gate failed: " + ", ".join(failed),
        unresolved_conflicts=[str(parsed.get("conflict_summary", ""))] if has_conflict else [],
        required_next_actions=["Resolve or disclose conflicts and state uncertainty boundaries."],
        revised_answer_hint=required_revision or "Revise the answer to disclose uncertainty and conflicts.",
        confidence=0.7,
        layer_results=[{"layer": "conflict_uncertainty", **parsed}],
    )


def _needs_uncertainty(task: RAESTask) -> bool:
    text = f"{task.task_type} {task.question}".lower()
    tokens = ("current", "latest", "as of", "today", "now", "verify", "conflict", "status", "leaderboard")
    return task.task_type in UNCERTAINTY_TASK_TYPES or any(token in text for token in tokens)


def _has_uncertainty_language(answer: str) -> bool:
    lowered = answer.lower()
    tokens = (
        "as of",
        "accessed",
        "current",
        "version",
        "uncertain",
        "not enough evidence",
        "insufficient evidence",
        "conflict",
        "appears",
        "likely",
    )
    return any(token in lowered for token in tokens)
