"""
The agentic query loop — the heart of the search agent.

A while(true) loop with explicit State,
streaming via AsyncGenerator, tool execution, compaction, and stop hooks.

The loop:
  1. Check guards (max_turns)
  2. Compact context if too large
  3. Call LLM via streaming
  4. If no tool calls → run stop hooks → break or inject feedback
  5. Execute tools (parallel for concurrency-safe ones)
  6. Inject tool results + citations → continue

Interactive tools (ask_user):
  The generator is bidirectional — it yields StreamEvents and receives
  user answers via asend(). When a tool returns a "pending_question"
  in its metadata, the loop yields a USER_QUESTION event and the
  caller asend()s the user's answer string. This avoids Futures,
  deadlocks, and background tasks.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator

from src.core.answer_quality import final_answer_quality_issue
from src.core.errors import ApiQuotaExceeded, quota_error_from_exception
from src.core.tool import ToolRegistry, ToolUseContext
from src.core.types import (
    ContentBlock,
    EventType,
    LoopState,
    Message,
    StreamEvent,
    ToolResult,
)
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class QueryParams:
    """
    Parameters for a single query loop invocation.
    Runs a complete research session with tool use and compaction.
    """
    query: str
    system_prompt: str
    tool_registry: ToolRegistry
    llm_client: LLMClient

    # Existing conversation history (empty for new sessions)
    history: list[Message] = field(default_factory=list)

    # Guards
    max_turns: int = 20
    max_search: int = 20  # Max search tool calls (web, academic, news)
    max_fetch: int = 20   # Max web_fetch tool calls

    # Compaction
    compact_threshold_tokens: int = 80000

    # Session tracking
    session_id: str = ""

    # Hook engine (injected, optional)
    hook_engine: object | None = None

    # Rate limiter (injected, optional)
    rate_limiter: object | None = None

    # Cache directory for oversized tool results
    cache_dir: str = "./cache"

    # Max characters to include in streamed tool_result events. The web UI
    # keeps the default preview; RAES trajectory runs can raise this to retain
    # full-enough observations for training export.
    tool_result_event_max_chars: int = 500

    # Optional first-turn tool forcing for models that otherwise answer from
    # memory before entering the research loop.
    force_first_tool: str = ""

    # Execute one real web_search before the first LLM turn. This is more
    # reliable than API-level tool_choice for local OpenAI-compatible servers
    # that may not handle forced streaming tool calls correctly.
    prefill_search: bool = False


async def query_loop(params: QueryParams) -> AsyncGenerator[StreamEvent, str | None]:
    """
    The main agentic loop. Streams events to the caller (WebSocket handler).

    This is a bidirectional AsyncGenerator — the caller iterates events
    and can send values back via asend() for interactive tools (ask_user).
    Normal events yield None; USER_QUESTION events receive the user's
    answer string.
    - while(true) with explicit LoopState
    - Guards: max_turns
    - Compact before each LLM call if context is too large
    - Parallel tool execution for concurrency-safe tools
    - Stop hooks as quality gates before finalizing
    """
    # --- Initialize state ---
    state = LoopState(
        messages=list(params.history),
        turn_count=0,
        citations=[],
    )

    # Reset Responses API chain for a fresh session
    if hasattr(params.llm_client, "reset_response_chain"):
        params.llm_client.reset_response_chain(session_id=params.session_id)

    # Add the user query as the first message (if not already in history)
    if not state.messages or state.messages[-1].role != "user":
        state.messages.append(Message(role="user", content=params.query))

    # Build tool schemas for the LLM
    tool_schemas = params.tool_registry.get_api_schemas()
    concurrent_safe = params.tool_registry.get_concurrent_safe()

    yield StreamEvent(
        type=EventType.STATUS,
        data={"message": "Research started"},
    )

    if params.prefill_search and tool_schemas:
        prefill_call = {
            "tool_use_id": f"prefill_{uuid.uuid4().hex[:12]}",
            "tool_name": "web_search",
            "tool_input": {"query": params.query},
        }
        yield StreamEvent(type=EventType.TOOL_USE, data=prefill_call)
        prefill_results = await _execute_tools(
            tool_calls=[prefill_call],
            registry=params.tool_registry,
            state=state,
            params=params,
            concurrent_safe=concurrent_safe,
        )
        prefill_result = prefill_results[0] if prefill_results else ToolResult(data="", is_error=True)
        prefill_text = prefill_result.data or ""
        result_limit = max(0, params.tool_result_event_max_chars)
        event_result = prefill_text[:result_limit] if result_limit else ""
        yield StreamEvent(
            type=EventType.TOOL_RESULT,
            data={
                "tool_use_id": prefill_call["tool_use_id"],
                "tool_name": prefill_call["tool_name"],
                "result": event_result,
                "result_chars": len(prefill_text),
                "result_truncated_for_event": len(prefill_text) > len(event_result),
                "is_error": prefill_result.is_error,
                "truncated": prefill_result.truncated,
                "cached_path": prefill_result.cached_path,
            },
        )
        evidence_note = (
            "\n\nA preliminary web_search has already been executed for this question. "
            "Use these retrieved results as evidence context, then continue with additional "
            "web_fetch, deep_read, cite_source, or search calls if needed before the final answer.\n\n"
            f"Preliminary web_search result:\n{prefill_text}"
        )
        if state.messages and state.messages[-1].role == "user" and isinstance(state.messages[-1].content, str):
            state.messages[-1].content += evidence_note
        else:
            state.messages.append(Message(role="user", content=params.query + evidence_note))
        state.search_count += 1
        for citation in prefill_result.citations:
            state.citations.append(citation)
            yield StreamEvent(
                type=EventType.CITATION,
                data=citation.to_dict(),
            )

    # Safety valve: if all tool calls are skipped (over-limit) for N
    # consecutive turns, force _final_answer to prevent infinite loops.
    _consecutive_all_skipped = 0
    _force_final = False

    # --- Main loop ---
    while True:
        state.turn_count += 1

        # --- Guard: max turns ---
        if state.turn_count > params.max_turns:
            yield StreamEvent(
                type=EventType.STATUS,
                data={"message": f"Reached maximum turns ({params.max_turns}). Synthesizing final answer..."},
            )
            # Give the LLM one last chance to answer (no tools)
            async for ev in _final_answer(state, params):
                yield ev
            break

        # --- Guard: per-tool limits ---
        # For Responses API (GPT models), we preserve the server-side chain
        # by continuing the loop and letting per-call filtering inject dummy
        # results.  Only force _final_answer when BOTH limits are hit or
        # when using Chat Completions.
        _use_responses = params.llm_client.uses_responses_api

        if state.search_count >= params.max_search and state.fetch_count >= params.max_fetch:
            if not _use_responses:
                # Chat Completions: force final answer
                yield StreamEvent(
                    type=EventType.STATUS,
                    data={"message": "Reached search and fetch limits. Synthesizing final answer..."},
                )
                async for ev in _final_answer(state, params):
                    yield ev
                break
            # Responses API: continue loop, safety valve will handle exit

        # Log individual limit warnings (but don't terminate — the agent
        # can still use the other tool type)
        if state.search_count >= params.max_search and not _use_responses:
            if state.search_count == params.max_search:  # Log once
                yield StreamEvent(
                    type=EventType.STATUS,
                    data={"message": f"Search limit reached ({params.max_search}). Continuing with fetch tools..."},
                )

        if state.fetch_count >= params.max_fetch and not _use_responses:
            if state.fetch_count == params.max_fetch:  # Log once
                yield StreamEvent(
                    type=EventType.STATUS,
                    data={"message": f"Fetch limit reached ({params.max_fetch}). Continuing with search tools..."},
                )

        # --- Compaction ---
        # Skip for Responses API models: the server manages context via
        # previous_response_id + truncation:"auto".  Local compaction would
        # delete messages that the server-side chain still references,
        # causing "No tool output found" errors.
        if not _use_responses:
            try:
                from src.core.compact import should_compact, compact_messages
                if should_compact(state.messages, params.compact_threshold_tokens):
                    yield StreamEvent(
                        type=EventType.STATUS,
                        data={"message": "Compacting context..."},
                    )
                    state.messages = await compact_messages(
                        state.messages,
                        params.compact_threshold_tokens,
                    )
                    state.compaction_count += 1
                    yield StreamEvent(
                        type=EventType.STATUS,
                        data={"message": f"Context compacted (#{state.compaction_count})"},
                    )
            except ImportError:
                pass  # Compaction not yet implemented — skip

        # --- Convert messages to API format ---
        api_messages = [msg.to_api_dict() for msg in state.messages]

        # --- Call LLM ---
        tool_calls: list[dict] = []
        assistant_text_parts: list[str] = []
        assistant_content_blocks: list[ContentBlock] = []
        llm_error = False  # Track API errors to skip stop hooks

        try:
            forced_tool_choice = None
            if state.turn_count == 1 and params.force_first_tool and tool_schemas:
                forced_tool_choice = {
                    "type": "function",
                    "function": {"name": params.force_first_tool},
                }
            async for event in params.llm_client.stream(
                messages=api_messages,
                system_prompt=params.system_prompt,
                tools=tool_schemas if tool_schemas else None,
                tool_choice=forced_tool_choice,
                max_tokens=params.llm_client.config.max_tokens,
                session_id=params.session_id,
            ):
                # Pass through to caller (streams to WebSocket)
                yield event

                # Accumulate for state management
                if event.type == EventType.TEXT_DELTA:
                    text = event.data.get("text", "")
                    assistant_text_parts.append(text)

                elif event.type == EventType.TOOL_USE:
                    tool_calls.append(event.data)

                elif event.type == EventType.ERROR:
                    # LLM error — stop the loop
                    logger.error(f"LLM error: {event.data}")
                    llm_error = True
                    break

        except ApiQuotaExceeded:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLM stream: {e}")
            yield StreamEvent(
                type=EventType.ERROR,
                data={
                    "message": f"LLM stream error: {str(e)}",
                    "traceback": traceback.format_exc(),
                },
            )
            break

        # Skip stop hooks when the last turn was an API error —
        # hooks evaluating an empty response create a death spiral:
        # error → hook "no answer yet" → retry → same error → …
        if llm_error:
            logger.warning("LLM error occurred, breaking loop (skipping stop hooks)")
            break

        # --- Build assistant message for state ---
        full_text = "".join(assistant_text_parts)

        if tool_calls and full_text:
            # Assistant produced both text and tool calls
            assistant_content_blocks.append(
                ContentBlock(type="text", text=full_text)
            )
            for tc in tool_calls:
                assistant_content_blocks.append(ContentBlock(
                    type="tool_use",
                    tool_use_id=tc["tool_use_id"],
                    tool_name=tc["tool_name"],
                    tool_input=tc["tool_input"],
                ))
            state.messages.append(Message(
                role="assistant",
                content=assistant_content_blocks,
            ))
        elif tool_calls:
            # Only tool calls (no text)
            for tc in tool_calls:
                assistant_content_blocks.append(ContentBlock(
                    type="tool_use",
                    tool_use_id=tc["tool_use_id"],
                    tool_name=tc["tool_name"],
                    tool_input=tc["tool_input"],
                ))
            state.messages.append(Message(
                role="assistant",
                content=assistant_content_blocks,
            ))
        elif full_text:
            # Only text (no tool calls)
            state.messages.append(Message(
                role="assistant",
                content=full_text,
            ))

        # --- No tool calls → model wants to stop ---
        if not tool_calls:
            # Run stop hooks (quality gate)
            should_continue, feedback, hook_metadata = await _run_stop_hooks(state, params)
            if hook_metadata:
                yield StreamEvent(
                    type=EventType.RAES_GATE,
                    data=hook_metadata,
                )
            if should_continue and feedback:
                # Hook says answer isn't good enough — inject feedback
                yield StreamEvent(
                    type=EventType.STATUS,
                    data={"message": f"Quality check: {feedback}"},
                )
                state.messages.append(Message(
                    role="user",
                    content=feedback,
                ))
                continue

            # All hooks passed (or no hooks) — finalize
            break

        # --- Execute tool calls ---
        # Filter out tool calls that would exceed per-tool limits.
        # The LLM may issue multiple tool calls in one turn, so we must
        # enforce limits per-call, not just per-turn.
        allowed_tool_calls: list[dict] = []
        skipped_tool_calls: list[dict] = []
        _pending_search = state.search_count
        _pending_fetch = state.fetch_count
        for tc in tool_calls:
            name = tc["tool_name"]
            if name in ("web_search", "academic_search", "news_search"):
                if _pending_search >= params.max_search:
                    skipped_tool_calls.append(tc)
                    continue
                _pending_search += 1
            elif name == "web_fetch":
                if _pending_fetch >= params.max_fetch:
                    skipped_tool_calls.append(tc)
                    continue
                _pending_fetch += 1
            allowed_tool_calls.append(tc)

        yield StreamEvent(
            type=EventType.STATUS,
            data={"message": f"Executing {len(allowed_tool_calls)} tool(s)..."},
        )

        tool_results = await _execute_tools(
            tool_calls=allowed_tool_calls,
            registry=params.tool_registry,
            state=state,
            params=params,
            concurrent_safe=concurrent_safe,
        )

        # Add "limit reached" results for skipped tool calls
        for tc in skipped_tool_calls:
            name = tc["tool_name"]
            if name in ("web_search", "academic_search", "news_search"):
                msg = "Search limit reached. You cannot perform more searches."
            elif name == "web_fetch":
                msg = "Fetch limit reached. You cannot fetch more pages."
            else:
                msg = f"{name} limit reached."
            tool_results.append(ToolResult(
                data=msg,
                is_error=False,
            ))
            allowed_tool_calls.append(tc)  # re-add so zip() below pairs correctly

        # Safety valve: if ALL tool calls were skipped for too many
        # consecutive turns, the model is stuck calling over-limit tools.
        _real_call_count = len(allowed_tool_calls) - len(skipped_tool_calls)
        if _real_call_count == 0 and skipped_tool_calls:
            _consecutive_all_skipped += 1
            if _consecutive_all_skipped >= 3:
                yield StreamEvent(
                    type=EventType.STATUS,
                    data={"message": "All tools at limit for 3 turns. Synthesizing final answer..."},
                )
                if not _use_responses:
                    async for ev in _final_answer(state, params):
                        yield ev
                    break
                # Responses API: let dummy results get injected below,
                # then add a user nudge. Next iteration will call LLM
                # with no tools, preserving the chain.
                _force_final = True
        else:
            _consecutive_all_skipped = 0

        # --- Inject tool results into conversation ---
        # For OpenAI-compatible APIs, tool results go as separate messages
        # with role="tool" and the tool_call_id
        for tc, result in zip(allowed_tool_calls, tool_results):
            # Handle interactive tool (ask_user) — yield question to
            # the caller and receive the user's answer via asend().
            pending = result.metadata.get("pending_question")
            if pending:
                answer = yield StreamEvent(
                    type=EventType.USER_QUESTION,
                    data={
                        "tool_use_id": tc["tool_use_id"],
                        "question": pending["question"],
                        "options": pending["options"],
                    },
                )
                # Default to first option if no answer received
                if not answer:
                    answer = pending["options"][0]["label"] if pending["options"] else ""
                result = ToolResult(data=f"User answered: {answer}")

            # Stream result event to UI / trajectory recorder.
            result_text = result.data or ""
            result_limit = max(0, params.tool_result_event_max_chars)
            event_result = result_text[:result_limit] if result_limit else ""
            yield StreamEvent(
                type=EventType.TOOL_RESULT,
                data={
                    "tool_use_id": tc["tool_use_id"],
                    "tool_name": tc["tool_name"],
                    "result": event_result,
                    "result_chars": len(result_text),
                    "result_truncated_for_event": len(result_text) > len(event_result),
                    "is_error": result.is_error,
                    "truncated": result.truncated,
                    "cached_path": result.cached_path,
                },
            )

            # Add to conversation history
            state.messages.append(Message(
                role="tool",
                content=result.data,
                metadata={
                    "tool_call_id": tc["tool_use_id"],
                    "tool_name": tc["tool_name"],
                },
            ))

            # Track per-tool counts for limit guards
            tool_name = tc["tool_name"]
            if tool_name in ("web_search", "academic_search", "news_search"):
                state.search_count += 1
            elif tool_name == "web_fetch":
                state.fetch_count += 1

            # Accumulate citations
            for citation in result.citations:
                state.citations.append(citation)
                yield StreamEvent(
                    type=EventType.CITATION,
                    data=citation.to_dict(),
                )

        # Emit plan_update event if a research plan exists (tool may have modified it)
        if state.research_plan is not None:
            yield StreamEvent(
                type=EventType.PLAN_UPDATE,
                data=state.research_plan.to_dict(),
            )

        # Safety valve: after injecting dummy results, nudge the model
        # to produce a final answer on the next iteration (no tools).
        if _force_final:
            state.messages.append(Message(
                role="user",
                content=(
                    "All tool limits have been reached. You cannot make any more tool calls. "
                    "Please provide your final answer now based on all the research you have gathered."
                ),
            ))
            tool_schemas = None
            continue

        # --- Soft nudge: suggest research_plan if not yet used after several searches ---
        # Only nudge once (check via transition_reason marker)
        already_nudged = any(
            "plan_nudge" in (m.metadata.get("_tag", "") or "")
            for m in state.messages
        )
        if not already_nudged and params.tool_registry.get("research_plan") is not None:
            search_count = sum(
                1 for m in state.messages
                if m.role == "tool" and m.metadata.get("tool_name") in ("web_search", "academic_search", "news_search")
            )
            if state.research_plan is None and search_count >= 3:
                state.messages.append(Message(
                    role="user",
                    content=(
                        "You've done several searches without creating a research plan. "
                        "This query appears to have multiple aspects. Please use "
                        "research_plan(action='create') now to organize your remaining "
                        "research into sub-tasks before continuing."
                    ),
                    metadata={"_tag": "plan_nudge"},
                ))

    # --- Finalize ---
    # Build session summary for post-session memory extraction
    final_answer = state.last_assistant_message or ""
    plan_findings = ""
    if state.research_plan and state.research_plan.tasks:
        plan_findings = "\n".join(
            f"- {t.title}: {t.findings}"
            for t in state.research_plan.tasks
            if t.findings
        )

    yield StreamEvent(
        type=EventType.STATUS,
        data={
            "message": f"Research complete. {len(state.citations)} sources cited. "
                       f"Turns: {state.turn_count}.",
        },
    )

    # Condense messages for conversation continuity across turns.
    # Only user messages and assistant text are kept — tool messages
    # (research mechanics) are dropped to save context tokens.
    condensed_history = _condense_for_history(state.messages)

    yield StreamEvent(
        type=EventType.DONE,
        data={
            "citations": [c.to_dict() for c in state.citations],
            "turn_count": state.turn_count,
            "compaction_count": state.compaction_count,
            "session_summary": {
                "query": params.query,
                "final_answer": final_answer,
                "plan_findings": plan_findings,
            },
            # Condensed history for the next turn's conversation continuity.
            # Serialized to dicts so the DONE event is JSON-serializable
            # (ws.send_json would fail on raw Message objects).
            "final_messages": [
                {"role": m.role, "content": m.text_content}
                for m in condensed_history
            ],
        },
    )


async def _execute_tools(
    tool_calls: list[dict],
    registry: ToolRegistry,
    state: LoopState,
    params: QueryParams,
    concurrent_safe: set[str],
) -> list[ToolResult]:
    """
    Execute tool calls, running concurrency-safe tools in parallel.

    Tools marked is_concurrency_safe can run simultaneously (e.g.,
    multiple web searches), while unsafe tools run sequentially.
    """
    context = ToolUseContext(
        session_id=params.session_id,
        turn_count=state.turn_count,
        cache_dir=Path(params.cache_dir),
        extra={
            "loop_state": state,
            "research_query": _extract_research_query(state.messages),
        },
        rate_limiter=params.rate_limiter,
    )

    # Separate into parallel-safe and sequential, tracking original indices
    parallel_indices: list[int] = []
    sequential_indices: list[int] = []

    for i, tc in enumerate(tool_calls):
        tool_name = tc["tool_name"]
        if tool_name in concurrent_safe:
            parallel_indices.append(i)
        else:
            sequential_indices.append(i)

    # Pre-allocate results list in tool_calls order
    results: list[ToolResult] = [ToolResult(data="", is_error=True) for _ in range(len(tool_calls))]

    # Run parallel-safe tools concurrently
    if parallel_indices:
        async def _run_one(tc: dict) -> ToolResult:
            return await _execute_single_tool(tc, registry, context)

        parallel_results = await asyncio.gather(
            *[_run_one(tool_calls[i]) for i in parallel_indices],
            return_exceptions=True,
        )
        for idx, result in zip(parallel_indices, parallel_results):
            if isinstance(result, Exception):
                if isinstance(result, ApiQuotaExceeded):
                    raise result
                logger.error(f"Tool {tool_calls[idx]['tool_name']} failed: {result}")
                results[idx] = ToolResult(
                    data=f"Tool execution failed: {str(result)}",
                    is_error=True,
                )
            else:
                results[idx] = result

    # Run sequential tools one at a time
    for idx in sequential_indices:
        result = await _execute_single_tool(tool_calls[idx], registry, context)
        results[idx] = result

    return results


async def _execute_single_tool(
    tc: dict,
    registry: ToolRegistry,
    context: ToolUseContext,
) -> ToolResult:
    """Execute a single tool call with validation and error handling."""
    tool_name = tc["tool_name"]
    tool_input = tc.get("tool_input", {})

    tool = registry.get(tool_name)
    if tool is None:
        logger.warning(f"Unknown tool: {tool_name}")
        return ToolResult(
            data=f"Error: Unknown tool '{tool_name}'. Available tools: "
                 f"{', '.join(t.name for t in registry.all_tools())}",
            is_error=True,
        )

    # Validate input
    validation = tool.validate_input(tool_input)
    if not validation.valid:
        logger.warning(f"Tool {tool_name} input validation failed: {validation.message}")
        return ToolResult(
            data=f"Invalid input for {tool_name}: {validation.message}",
            is_error=True,
        )

    # Execute tool calls. Quota/auth/rate-limit API failures are fatal for
    # batch runs and must not be hidden behind fallback behavior.
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            result = await tool.call(tool_input, context)
            logger.info(
                f"Tool {tool_name} completed: {len(result.data)} chars, "
                f"{len(result.citations)} citations, truncated={result.truncated}"
            )
            return result
        except ApiQuotaExceeded:
            raise
        except Exception as e:
            quota_error = quota_error_from_exception(tool_name, e)
            if quota_error is not None:
                raise quota_error
            error_str = str(e)
            # Retry transient non-quota 429-like tool errors with exponential backoff.
            if "429" in error_str and attempt < max_retries:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    f"Tool {tool_name} rate-limited (429), retrying in {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait)
                continue
            # All other errors or retries exhausted — return friendly message
            logger.error(f"Tool {tool_name} execution error: {e}", exc_info=True)
            return ToolResult(
                data=f"The {tool_name} service is temporarily unavailable. Please try again later.",
                is_error=False,
            )


async def _final_answer(
    state: LoopState,
    params: QueryParams,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Make one last LLM call without tools to force a final answer.

    Called when a guard (max_turns) fires while the agent is
    still mid-research. We build clean messages (no tool_calls/tool roles)
    to avoid Anthropic API errors, then ask the LLM to synthesize.
    """
    # Build clean messages.  Do not carry assistant scratchpad/status text
    # such as "Let me fetch..." into the final-only call; weak models can
    # imitate that style and stop before producing the actual answer.
    # This avoids Anthropic's "tools= param required" error when the
    # conversation contains tool_use blocks but no tools are provided.
    tool_findings = []
    original_question = params.query

    for msg in state.messages:
        if msg.role == "user" and not msg.metadata.get("_tag"):
            original_question = original_question or msg.text_content
        elif msg.role == "tool":
            # Collect tool results as context
            tool_name = msg.metadata.get("tool_name", "tool")
            content = msg.text_content[:1200]  # Truncate to avoid huge context
            if content.strip():
                tool_findings.append(f"[{tool_name}]: {content}")

    def build_clean_messages(retry_issue: str | None = None, previous_text: str = "") -> list[dict[str, str]]:
        synthesis_msg = f"Original question:\n{original_question}\n\n"
        if tool_findings:
            # Keep only the last 12 results to avoid overwhelming context,
            # but give the model enough evidence to write a grounded answer.
            recent = tool_findings[-12:]
            synthesis_msg += "Research evidence already gathered:\n\n"
            synthesis_msg += "\n\n".join(recent)
            synthesis_msg += "\n\n---\n\n"

        if retry_issue:
            synthesis_msg += (
                "Your previous response was not an acceptable final answer "
                f"({retry_issue}). Do not continue searching or describe what you will do next.\n"
            )
            if previous_text.strip():
                synthesis_msg += f"Previous unacceptable response:\n{previous_text.strip()[:800]}\n\n"

        synthesis_msg += (
            "You have reached the tool limit and cannot make any more tool calls. "
            "Write the final answer now using only the evidence above. "
            "Do not say 'let me', 'I will', 'I need to check', or describe next steps. "
            "Start directly with the answer or conclusion. "
            "If the evidence is insufficient, explicitly say that public evidence is insufficient "
            "and explain what is known, what is missing, and why uncertainty is required. "
            "Include source names or URLs from the gathered evidence where possible."
        )
        return [{"role": "user", "content": synthesis_msg}]

    try:
        # For Responses API (GPT models): instead of resetting the chain
        # and losing all research context, send dummy outputs for any
        # pending function calls through the existing chain, then the
        # synthesis message.  This preserves the full conversation state.
        use_responses_api = params.llm_client.uses_responses_api

        # Use a smaller max_tokens for final synthesis to avoid long waits
        final_max_tokens = min(params.llm_client.config.max_tokens, 16384)

        last_text = ""
        last_issue: str | None = None
        for attempt in range(2):
            final_text_parts: list[str] = []
            clean_messages = build_clean_messages(last_issue, last_text) if attempt else build_clean_messages()
            # Reset chain and use clean messages with no tools.  This prevents
            # the model from continuing a pending tool-calling pattern.
            if hasattr(params.llm_client, "reset_response_chain"):
                params.llm_client.reset_response_chain(session_id=params.session_id)
            async for event in params.llm_client.stream(
                messages=clean_messages,
                system_prompt=params.system_prompt,
                tools=None,  # No tools — force a text-only response
                max_tokens=final_max_tokens,
                session_id=params.session_id,
            ):
                if event.type == EventType.TEXT_DELTA:
                    final_text_parts.append(event.data.get("text", ""))
                yield event

            last_text = "".join(final_text_parts)
            last_issue = final_answer_quality_issue(last_text)
            if not last_issue:
                state.messages.append(Message(role="assistant", content=last_text))
                return
            if attempt == 0:
                yield StreamEvent(
                    type=EventType.STATUS,
                    data={"message": f"Final answer looked incomplete ({last_issue}); retrying final synthesis..."},
                )

        # Append the final attempt even if imperfect so downstream quality
        # checks can mark the row accurately.
        if last_text.strip():
            state.messages.append(Message(role="assistant", content=last_text))
    except ApiQuotaExceeded:
        raise
    except Exception as e:
        logger.error(f"Final answer LLM error: {e}")
        yield StreamEvent(
            type=EventType.ERROR,
            data={"message": f"Failed to generate final answer: {str(e)}"},
        )


async def _run_stop_hooks(
    state: LoopState,
    params: QueryParams,
) -> tuple[bool, str | None, dict | None]:
    """
    Run stop hooks (quality gates) before finalizing the answer.

    Returns (should_continue, feedback, metadata).
    If should_continue is True, the loop injects feedback and continues.

    Follows the stop hooks pattern — the model thinks it's done,
    but a quality check might disagree and force another iteration.
    """
    if params.hook_engine is None:
        return False, None, None

    try:
        # Hook engine should implement run_stop_hooks(state) -> HookResult
        hook_engine = params.hook_engine
        if hasattr(hook_engine, "run_stop_hooks"):
            result = await hook_engine.run_stop_hooks(state)
            return (
                result.should_continue,
                getattr(result, "feedback", None),
                getattr(result, "metadata", None),
            )
    except ApiQuotaExceeded:
        raise
    except Exception as e:
        logger.warning(f"Stop hook error (ignoring): {e}")

    return False, None, None


def _extract_research_query(messages: list[Message]) -> str:
    """
    Extract the original research query from the conversation messages.

    Returns the first real user message (skipping system injections like
    plan_nudge). This is used by tools like web_fetch that need the
    research question for content extraction relevance filtering.
    """
    for msg in messages:
        if msg.role == "user" and not msg.metadata.get("_tag"):
            return msg.text_content
    return ""


def _condense_for_history(messages: list[Message]) -> list[Message]:
    """
    Condense loop messages into a compact history for the next turn.

    Keeps user messages and assistant text responses.
    Drops tool-call details and tool results to save context tokens.
    Preserves the conversation flow without the research mechanics.

    Messages accumulate across turns, but we strip out the tool
    interaction details to keep the history lean.
    """
    condensed = []
    for msg in messages:
        if msg.role == "user":
            # Keep user messages but skip system injections (plan_nudge, etc.)
            if not msg.metadata.get("_tag"):
                condensed.append(msg)
        elif msg.role == "assistant":
            # Keep only the text content, drop tool_calls
            text = msg.text_content.strip()
            if text:
                condensed.append(Message(role="assistant", content=text))
        # Skip tool messages entirely — they're research mechanics,
        # not conversational context
    return condensed
