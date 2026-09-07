"""RAES-Gate stop-condition controller."""

from __future__ import annotations

from typing import Any

from src.raes_eval.schemas import AgentTrace, GateDecision


def aggregate_gate_decisions(
    decisions: list[GateDecision],
    trace: AgentTrace,
    budget: dict[str, Any] | None = None,
) -> GateDecision:
    if not decisions:
        return GateDecision.allow()

    budget = budget or {}
    exhausted = bool(budget.get("exhausted", False))
    failed_checks = []
    missing_facets = []
    unsupported_claims = []
    unresolved_conflicts = []
    required_next_actions = []
    suggested_queries = []
    layer_results = []

    for decision in decisions:
        failed_checks.extend(decision.failed_checks)
        missing_facets.extend(decision.missing_facets)
        unsupported_claims.extend(decision.unsupported_claims)
        unresolved_conflicts.extend(item for item in decision.unresolved_conflicts if item)
        required_next_actions.extend(decision.required_next_actions)
        suggested_queries.extend(decision.suggested_queries)
        layer_results.extend(decision.layer_results)

    if exhausted:
        if any(check in failed_checks for check in ("contradicted_claim", "unsupported_claim", "claims_without_citations")):
            return _combined(
                "revise_answer",
                failed_checks,
                "RAES-Gate budget is exhausted and claim support is weak; remove unsupported claims and answer only what the evidence supports.",
                missing_facets,
                unsupported_claims,
                unresolved_conflicts,
                required_next_actions,
                suggested_queries,
                layer_results,
                hint="Revise conservatively. Remove unsupported claims, cite only supported facts, and state uncertainty.",
            )
        if unresolved_conflicts:
            return _combined(
                "add_uncertainty",
                failed_checks,
                "RAES-Gate budget is exhausted with unresolved conflicts; final answer must disclose the conflict.",
                missing_facets,
                unsupported_claims,
                unresolved_conflicts,
                required_next_actions,
                suggested_queries,
                layer_results,
                hint="Provide a final answer with explicit conflict disclosure and uncertainty boundaries.",
            )
        return _combined(
            "add_uncertainty",
            failed_checks,
            "RAES-Gate budget is exhausted before all checks passed; final answer must be uncertainty-calibrated.",
            missing_facets,
            unsupported_claims,
            unresolved_conflicts,
            required_next_actions,
            suggested_queries,
            layer_results,
            hint="Give a cautious final answer, state evidence limits, and avoid unsupported certainty.",
        )

    priority = [
        ("contradicted_claim", "revise_answer"),
        ("insufficient_source_count", "continue_search"),
        ("insufficient_independent_domains", "continue_search"),
        ("missing_primary_source", "continue_search"),
        ("missing_recent_source", "continue_search"),
        ("missing_repository_or_documentation_source", "continue_search"),
        ("missing_academic_or_official_source", "continue_search"),
        ("missing_critical_facet", "continue_search"),
        ("unresolved_conflict", "revise_answer"),
        ("missing_uncertainty_statement", "add_uncertainty"),
        ("unsupported_claim", "revise_answer"),
        ("claims_without_citations", "revise_answer"),
        ("draft_too_short", "revise_answer"),
    ]
    selected_decision = decisions[0].decision
    for check, action in priority:
        if check in failed_checks:
            selected_decision = action
            break

    return _combined(
        selected_decision,
        failed_checks,
        "RAES-Gate failed: " + ", ".join(dict.fromkeys(failed_checks)),
        missing_facets,
        unsupported_claims,
        unresolved_conflicts,
        required_next_actions,
        suggested_queries,
        layer_results,
        hint=_revision_hint(selected_decision, failed_checks),
    )


def _combined(
    decision: str,
    failed_checks: list[str],
    reason: str,
    missing_facets: list[str],
    unsupported_claims: list[str],
    unresolved_conflicts: list[str],
    required_next_actions: list[str],
    suggested_queries: list[str],
    layer_results: list[dict[str, Any]],
    *,
    hint: str | None = None,
) -> GateDecision:
    return GateDecision(
        decision=decision,
        passed=False,
        failed_checks=list(dict.fromkeys(failed_checks)),
        reason=reason,
        missing_facets=list(dict.fromkeys(missing_facets)),
        unsupported_claims=list(dict.fromkeys(unsupported_claims)),
        unresolved_conflicts=list(dict.fromkeys(unresolved_conflicts)),
        required_next_actions=list(dict.fromkeys(required_next_actions)),
        suggested_queries=list(dict.fromkeys(suggested_queries))[:5],
        revised_answer_hint=hint,
        confidence=0.7,
        layer_results=layer_results,
    )


def _revision_hint(decision: str, failed_checks: list[str]) -> str:
    if decision == "continue_search":
        return "Continue searching/fetching independent authoritative sources, then cite them in the revised answer."
    if decision == "add_uncertainty":
        return "Revise the answer to include uncertainty, access-time/version boundaries, and conflict disclosure where relevant."
    if "contradicted_claim" in failed_checks:
        return "Remove or rewrite contradicted claims and explain the conflict with citations."
    return "Revise the answer so every factual claim is supported by cited evidence."
