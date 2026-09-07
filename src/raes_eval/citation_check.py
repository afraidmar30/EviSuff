"""Claim-citation support checks for RAES-Gate."""

from __future__ import annotations

import json
import re
from typing import Any

from src.raes_eval.gate_prompts import CITATION_SUPPORT_PROMPT, CLAIM_EXTRACTION_PROMPT
from src.raes_eval.gate_utils import json_side_query, source_summaries
from src.raes_eval.schemas import AgentClaim, DraftAnswer, GateDecision, RetrievedSource


CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "text": {"type": "string"},
                    "citation_ids": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "string"},
                },
                "required": ["claim_id", "text", "citation_ids", "importance"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_support": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "support": {"type": "string"},
                    "supporting_source_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["claim_id", "support", "supporting_source_ids", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claim_support"],
    "additionalProperties": False,
}


async def extract_claims(draft: DraftAnswer, sources: list[RetrievedSource]) -> list[AgentClaim]:
    prompt = json.dumps(
        {
            "draft_answer": draft.text[:7000],
            "available_sources": source_summaries(sources, max_text_chars=300),
        },
        ensure_ascii=False,
    )
    parsed = await json_side_query(
        prompt=prompt,
        system=CLAIM_EXTRACTION_PROMPT,
        output_schema=CLAIM_SCHEMA,
        max_tokens=1200,
    )
    raw_claims = parsed.get("claims") if isinstance(parsed, dict) else None
    if isinstance(raw_claims, list) and raw_claims:
        return _normalize_claims(raw_claims)
    return fallback_extract_claims(draft, sources)


async def check_claim_citation_support(
    claims: list[AgentClaim],
    sources: list[RetrievedSource],
) -> GateDecision | None:
    if not claims:
        return None

    no_citation_claims = [claim for claim in claims if not claim.citation_ids]
    if not sources:
        return GateDecision(
            decision="continue_search",
            passed=False,
            failed_checks=["no_sources_for_claims"],
            reason="Claim-Citation Support Gate failed: factual claims have no retrieved sources.",
            unsupported_claims=[claim.text for claim in claims[:5]],
            required_next_actions=["Search for sources and cite claims with retrieved evidence."],
            confidence=0.85,
            layer_results=[{"layer": "claim_citation_support", "claims": [claim.to_dict() for claim in claims]}],
        )

    prompt = json.dumps(
        {
            "claims": [claim.to_dict() for claim in claims],
            "sources": source_summaries(sources, max_text_chars=900),
        },
        ensure_ascii=False,
    )
    parsed = await json_side_query(
        prompt=prompt,
        system=CITATION_SUPPORT_PROMPT,
        output_schema=SUPPORT_SCHEMA,
        max_tokens=1600,
    )

    if not isinstance(parsed, dict):
        if no_citation_claims:
            return GateDecision(
                decision="revise_answer",
                passed=False,
                failed_checks=["citation_verifier_unavailable", "claims_without_citations"],
                reason="Citation verifier failed; conservative fallback found factual claims without citations.",
                unsupported_claims=[claim.text for claim in no_citation_claims[:5]],
                required_next_actions=["Add citations to factual claims or remove unsupported claims."],
                confidence=0.45,
                layer_results=[{"layer": "claim_citation_support", "fallback": True}],
            )
        return None

    support_items = parsed.get("claim_support") or []
    contradicted = []
    unsupported = []
    no_citation = []
    partial = []
    claim_by_id = {claim.claim_id: claim for claim in claims}
    for item in support_items:
        claim_id = str(item.get("claim_id", ""))
        label = str(item.get("support", "")).lower()
        claim = claim_by_id.get(claim_id)
        text = claim.text if claim else claim_id
        if label == "contradicted":
            contradicted.append(text)
        elif label == "unsupported":
            unsupported.append(text)
        elif label == "no_citation":
            no_citation.append(text)
        elif label == "partially_supported":
            partial.append(text)

    if not contradicted and not unsupported and len(no_citation) <= 1 and len(partial) <= 2:
        return None

    failed = []
    if contradicted:
        failed.append("contradicted_claim")
    if unsupported:
        failed.append("unsupported_claim")
    if len(no_citation) > 1:
        failed.append("claims_without_citations")
    if len(partial) > 2:
        failed.append("many_partially_supported_claims")

    return GateDecision(
        decision="revise_answer",
        passed=False,
        failed_checks=failed,
        reason="Claim-Citation Support Gate failed: " + ", ".join(failed),
        unsupported_claims=(contradicted + unsupported + no_citation + partial)[:8],
        required_next_actions=[
            "Revise or remove unsupported claims.",
            "Add precise citations for factual claims.",
            "If sources contradict a claim, disclose the conflict instead of stating it as settled.",
        ],
        confidence=0.72,
        layer_results=[{"layer": "claim_citation_support", "claim_support": support_items}],
    )


def fallback_extract_claims(draft: DraftAnswer, sources: list[RetrievedSource]) -> list[AgentClaim]:
    source_ids = _source_ids_mentioned(draft.text, sources)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+", draft.text)
        if len(sentence.strip()) > 35
    ]
    claims = []
    for idx, sentence in enumerate(sentences[:10], 1):
        claims.append(AgentClaim(
            claim_id=f"c{idx}",
            text=sentence[:500],
            citation_ids=source_ids,
            importance="important",
        ))
    return claims


def _normalize_claims(raw_claims: list[dict[str, Any]]) -> list[AgentClaim]:
    claims = []
    for idx, claim in enumerate(raw_claims[:16], 1):
        text = str(claim.get("text", "")).strip()
        if not text:
            continue
        citation_ids = claim.get("citation_ids") or []
        if not isinstance(citation_ids, list):
            citation_ids = []
        importance = str(claim.get("importance", "important")).lower()
        if importance not in {"critical", "important", "minor"}:
            importance = "important"
        claims.append(AgentClaim(
            claim_id=str(claim.get("claim_id") or f"c{idx}"),
            text=text,
            citation_ids=[str(item) for item in citation_ids if item],
            importance=importance,
        ))
    return claims


def _source_ids_mentioned(text: str, sources: list[RetrievedSource]) -> list[str]:
    mentioned = []
    for source in sources:
        if source.url and source.url in text:
            mentioned.append(source.source_id)
        elif source.title and source.title[:40] in text:
            mentioned.append(source.source_id)
        elif source.cited:
            mentioned.append(source.source_id)
    return mentioned
