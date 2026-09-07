"""RAES-Gate stop hook for SearchClaw."""

from __future__ import annotations

import re
from typing import Any

from src.core.types import ContentBlock, LoopState, Message
from src.hooks.engine import Hook, HookEvaluation
from src.raes_eval.gate import RAESGate, RAESGateConfig
from src.raes_eval.schemas import AgentTrace, DraftAnswer, GateDecision, RAESTask, RetrievedSource


SEARCH_TOOLS = {"web_search", "academic_search", "news_search", "wechat_search"}


class RAESEvidenceGateHook(Hook):
    """Run the layered RAES-Gate before final answer finalization."""

    name = "raes_evidence_gate"
    description = "Schema-aware evidence sufficiency controller for RAES tasks"

    def __init__(
        self,
        *,
        task: RAESTask,
        config: RAESGateConfig | None = None,
    ) -> None:
        self.task = task
        self.task.assert_no_gold()
        self.config = config or RAESGateConfig()
        self.gate = RAESGate(self.config)
        self.iteration = 0
        self.decisions: list[dict[str, Any]] = []

    async def evaluate(self, state: LoopState, **kwargs: Any) -> HookEvaluation:
        self.iteration += 1
        draft = self._build_draft(state)
        trace = self._build_trace(state, draft)
        budget_exhausted = self.iteration >= self.config.max_gate_iterations

        decision = await self.gate.check(
            self.task,
            trace,
            draft,
            budget={
                "exhausted": budget_exhausted,
                "gate_iteration": self.iteration,
                "max_gate_iterations": self.config.max_gate_iterations,
            },
        )
        metadata = self._metadata(decision)
        self.decisions.append(metadata)

        if decision.passed:
            return HookEvaluation(passed=True, metadata=metadata)

        if self.iteration > self.config.max_gate_iterations:
            forced = dict(metadata)
            forced["decision"] = "final_with_uncertainty"
            forced["passed"] = True
            forced["forced_by_gate_budget"] = True
            forced["reason"] = (
                "RAES-Gate iteration budget exhausted. Allowing a cautious final answer "
                "after the previous revision request."
            )
            self.decisions.append(forced)
            return HookEvaluation(passed=True, metadata=forced)

        feedback = self._build_feedback(decision)
        if budget_exhausted:
            feedback += (
                "\n\nRAES-Gate has reached its configured iteration budget. "
                "Do not search indefinitely. Produce a conservative final answer on the next turn: "
                "remove unsupported claims, cite only supported facts, disclose conflicts, and state "
                "uncertainty or insufficiency where needed."
            )
        return HookEvaluation(passed=False, feedback=feedback, metadata=metadata)

    def _metadata(self, decision: GateDecision) -> dict[str, Any]:
        data = decision.to_dict()
        data.update({
            "hook": self.name,
            "task_id": self.task.id,
            "iteration": self.iteration,
            "max_gate_iterations": self.config.max_gate_iterations,
        })
        return data

    def _build_draft(self, state: LoopState) -> DraftAnswer:
        text = state.last_assistant_message or ""
        sources = self._sources_from_state(state)
        citation_ids = []
        for source in sources:
            if source.cited or (source.url and source.url in text) or (source.title and source.title[:40] in text):
                citation_ids.append(source.source_id)
        return DraftAnswer(text=text, citations=citation_ids)

    def _build_trace(self, state: LoopState, draft: DraftAnswer) -> AgentTrace:
        search_queries, fetched_urls = self._tool_activity(state.messages)
        return AgentTrace(
            task_id=self.task.id,
            question=self.task.question,
            search_queries=search_queries,
            retrieved_sources=self._sources_from_state(state, fetched_urls=fetched_urls),
            draft_answers=[draft],
            gate_decisions=list(self.decisions),
            final_answer=draft.text,
            final_citations=draft.citations,
            num_search_calls=state.search_count,
            num_fetch_calls=state.fetch_count,
        )

    def _sources_from_state(self, state: LoopState, fetched_urls: set[str] | None = None) -> list[RetrievedSource]:
        if fetched_urls is None:
            _, fetched_urls = self._tool_activity(state.messages)
        sources: list[RetrievedSource] = []
        seen: dict[str, RetrievedSource] = {}
        for citation in state.citations:
            url = str(citation.get("url", "") if isinstance(citation, dict) else getattr(citation, "url", ""))
            if not url:
                continue
            if url in seen:
                current = seen[url]
                snippet = str(citation.get("snippet", "") if isinstance(citation, dict) else getattr(citation, "snippet", ""))
                if len(snippet) > len(current.text):
                    current.text = snippet
                current.cited = current.cited or bool(citation.get("cited", False) if isinstance(citation, dict) else getattr(citation, "cited", False))
                current.was_fetched = current.was_fetched or url in fetched_urls
                continue
            source = RetrievedSource.from_citation(
                f"s{len(sources) + 1}",
                citation,
                was_fetched=url in fetched_urls,
            )
            sources.append(source)
            seen[url] = source
        return sources

    @staticmethod
    def _tool_activity(messages: list[Message]) -> tuple[list[str], set[str]]:
        search_queries: list[str] = []
        fetched_urls: set[str] = set()
        for message in messages:
            if message.role == "assistant" and isinstance(message.content, list):
                for block in message.content:
                    if not isinstance(block, ContentBlock) or block.type != "tool_use":
                        continue
                    tool_name = block.tool_name or ""
                    args = block.tool_input or {}
                    if tool_name in SEARCH_TOOLS:
                        query = str(args.get("query") or args.get("q") or "").strip()
                        if query:
                            search_queries.append(query)
                    elif tool_name == "web_fetch":
                        url = str(args.get("url") or "").strip()
                        if url:
                            fetched_urls.add(url)
            elif message.role == "tool" and message.metadata.get("tool_name") == "web_fetch":
                fetched_urls.update(re.findall(r"https?://[^\s)>\]]+", message.text_content))
        return search_queries, fetched_urls

    @staticmethod
    def _build_feedback(decision: GateDecision) -> str:
        lines = [
            "The RAES-Gate quality controller is not satisfied yet. Do not finalize.",
            f"Decision: {decision.decision}.",
            f"Reason: {decision.reason}",
        ]

        if decision.failed_checks:
            lines.append("Failed checks:")
            lines.extend(f"- {item}" for item in decision.failed_checks)
        if decision.missing_facets:
            lines.append("Missing or weak facets:")
            lines.extend(f"- {item}" for item in decision.missing_facets[:8])
        if decision.unsupported_claims:
            lines.append("Unsupported or weak claims:")
            lines.extend(f"- {item}" for item in decision.unsupported_claims[:8])
        if decision.unresolved_conflicts:
            lines.append("Unresolved conflicts:")
            lines.extend(f"- {item}" for item in decision.unresolved_conflicts[:5])
        if decision.required_next_actions:
            lines.append("Required next actions:")
            lines.extend(f"- {item}" for item in decision.required_next_actions[:8])
        if decision.suggested_queries:
            lines.append("Suggested next search queries:")
            lines.extend(f"- {query}" for query in decision.suggested_queries[:5])
        if decision.revised_answer_hint:
            lines.append(f"Revision hint: {decision.revised_answer_hint}")

        lines.append(
            "Continue with the requested action, then provide a revised answer with precise citations."
        )
        return "\n".join(lines)
