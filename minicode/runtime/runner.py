from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from minicode.core.messages import extract_text, extract_thinking_delta, make_human_message, marker_kind
from minicode.core.types import GraphEvent, RunTurnResult, ToolContext
from minicode.features.tools import ToolGraphAdapter
from minicode.platform.config import normalize_mode

from .graph_state import GraphState
from .model_factory import create_chat_model
from .prompts import (
    AFTER_TOOL_NUDGE,
    CONTINUE_NUDGE,
    EMPTY_NUDGE,
    assemble_system_prompt_split,
    find_latest_user_query,
)
from .retry import APIRetryExhaustedError, retry_with_backoff

_NUDGE_TEXTS = frozenset(
    {CONTINUE_NUDGE.strip(), EMPTY_NUDGE.strip(), AFTER_TOOL_NUDGE.strip()}
)


def _title_from_messages(messages: Iterable[BaseMessage]) -> str:
    for message in reversed(list(messages)):
        if getattr(message, "type", "") == "human":
            text = extract_text(message).strip()
            if text:
                return text[:120]
    return ""


def _make_event(events: list[GraphEvent], event_sink, kind: str, payload: dict[str, Any]) -> None:
    event = GraphEvent(kind=kind, payload=payload, timestamp=time.time())
    events.append(event)
    if event_sink is not None:
        event_sink(event)


def _extract_interrupt_payload(values: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = values.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", None)
    return value if isinstance(value, dict) else {"value": value}


def _fire_hook(services: Any, event: str, **data: Any) -> None:
    """Best-effort hook firing; swallow failures to never block the runner."""
    hooks = getattr(services, "hooks", None)
    if hooks is None:
        return
    try:
        hooks.fire(event, **data)
    except BaseException:
        pass


def run_turn(
    *,
    services: Any,
    prompt: str | None = None,
    messages: list[BaseMessage] | None = None,
    thread_id: str | None = None,
    resume: dict[str, Any] | None = None,
    mode: str | None = None,
    max_steps: int = 40,
    chat_model: Any | None = None,
    event_sink=None,
    persist: bool = True,
) -> RunTurnResult:
    services.migrator.migrate_once()
    resolved_mode = normalize_mode(mode or services.settings.auto_mode)
    input_messages = list(messages or [])
    if prompt:
        input_messages.append(HumanMessage(content=prompt))
    title = _title_from_messages(input_messages)
    resolved_thread_id = services.sessions.ensure_thread(thread_id, title=title)
    events: list[GraphEvent] = []

    # Pre-turn boundary checks.
    cost = getattr(services, "cost", None)
    if cost is not None:
        cost.begin_turn()
    permissions = getattr(services, "permissions", None)
    if permissions is not None and hasattr(permissions, "begin_turn"):
        permissions.begin_turn()
    auto = getattr(services, "auto", None)
    if auto is not None and hasattr(auto, "set_mode"):
        try:
            auto.set_mode(resolved_mode, changed_by="runtime")
        except ValueError:
            auto.set_mode("default", changed_by="runtime")
    if auto is not None and prompt:
        detected, reason = auto.detect_prompt_injection(prompt)
        if detected:
            _make_event(events, event_sink, "prompt_injection_detected", {"reason": reason})

    _fire_hook(
        services,
        "pre_turn",
        thread_id=resolved_thread_id,
        mode=resolved_mode,
        user_input=prompt or "",
    )

    model = chat_model or create_chat_model(services.settings, mode=resolved_mode)
    graph = _build_graph(
        services=services,
        chat_model=model,
        event_sink=lambda kind, payload: _make_event(events, event_sink, kind, payload),
    )
    config = {"configurable": {"thread_id": resolved_thread_id, "checkpoint_ns": ""}}
    latest_values: dict[str, Any] = {}
    interrupt_payload: dict[str, Any] | None = None

    try:
        if persist:
            context_manager = services.db.checkpointer()
        else:
            from contextlib import nullcontext

            context_manager = nullcontext(None)
        with context_manager as saver:
            compiled = graph.compile(checkpointer=saver)
            if resume is not None:
                graph_input: dict[str, Any] | Command = Command(resume=resume)
            else:
                graph_input = {
                    "messages": input_messages,
                    "route": "model_call",
                    "step_count": 0,
                    "max_steps": max_steps,
                    "saw_tool_result": False,
                    "empty_response_retries": 0,
                    "tool_error_count": 0,
                    "final_text": None,
                    "error": None,
                    "await_user": False,
                    "thread_id": resolved_thread_id,
                    "prompt_context": "",
                    "user_query": "",
                    "mode": resolved_mode,
                }
            for stream_item in compiled.stream(
                graph_input, config=config, stream_mode=["values", "custom"]
            ):
                if isinstance(stream_item, tuple) and len(stream_item) == 2:
                    stream_mode, payload = stream_item
                    if stream_mode == "values" and isinstance(payload, dict):
                        latest_values = dict(payload)
                        interrupt_payload = interrupt_payload or _extract_interrupt_payload(
                            payload
                        )
                    elif stream_mode == "custom" and isinstance(payload, dict):
                        _make_event(
                            events,
                            event_sink,
                            str(payload.get("kind", "custom")),
                            dict(payload.get("payload", {})),
                        )
            if persist:
                snapshot = compiled.get_state(config)
                if snapshot and getattr(snapshot, "values", None):
                    latest_values = dict(snapshot.values)
    except Exception as exc:  # noqa: BLE001
        error_text = f"{type(exc).__name__}: {exc}"
        _fire_hook(services, "on_error", thread_id=resolved_thread_id, error=error_text)
        return RunTurnResult(
            thread_id=resolved_thread_id,
            messages=(
                services.sessions.load_messages(resolved_thread_id)
                if persist
                else input_messages
            ),
            final_text=error_text,
            interrupt=None,
            await_user=False,
            error=error_text,
            events=events,
        )

    result_messages = list(latest_values.get("messages", []))
    final_text = latest_values.get("final_text")
    error_text = latest_values.get("error")
    await_user = bool(latest_values.get("await_user"))

    # Post-turn bookkeeping.
    if cost is not None:
        turn_cost = cost.end_turn()
        _make_event(events, event_sink, "turn_cost", {"usd": turn_cost})
    if permissions is not None and hasattr(permissions, "end_turn"):
        permissions.end_turn()
    if error_text:
        _fire_hook(services, "on_error", thread_id=resolved_thread_id, error=error_text)
    _fire_hook(
        services,
        "post_turn",
        thread_id=resolved_thread_id,
        final_text=final_text or "",
        await_user=await_user,
        error=error_text or "",
    )
    if auto is not None and isinstance(final_text, str) and final_text:
        unsafe, reason = auto.classify_output_safety(final_text)
        if unsafe:
            _make_event(events, event_sink, "unsafe_output_detected", {"reason": reason})

    return RunTurnResult(
        thread_id=resolved_thread_id,
        messages=result_messages,
        final_text=final_text,
        interrupt=interrupt_payload,
        await_user=await_user,
        error=error_text,
        events=events,
    )


# ---------------------------------------------------------------------------
# Anthropic prompt-caching helpers
#
# Caching on Anthropic is prefix-match: any byte change before a
# ``cache_control`` breakpoint invalidates everything after it. Render order
# is tools → system → messages, so a breakpoint on the last tool caches the
# tools array and a breakpoint on the static system block caches tools +
# static system together.
#
# Phase 2 extends caching into the message history. Dynamic content
# (memory / env / mode / skills / …) is moved OUT of the system message
# and shipped as a pseudo ``HumanMessage`` injected right before the
# current user query. The system prompt is then a pure static prefix that
# never varies turn-to-turn. Two additional breakpoints land inside the
# message list: one on the tail of prior-turn history (caches the whole
# conversation up to last turn) and one on the very last message of the
# current request (extends the cache across model_calls within the same
# turn). The pseudo message is NEVER persisted into ``state.messages`` —
# only the per-call prompt gets it — so history stays byte-stable.
# ---------------------------------------------------------------------------


def _bind_tools_with_caching(chat_model: Any, tool_schemas: list[dict[str, Any]], *, anthropic: bool) -> Any:
    """Bind tools, adding an Anthropic ``cache_control`` marker when possible.

    For OpenAI-compatible models we delegate to ``bind_tools`` unchanged —
    those providers cache automatically when the prefix stays stable. For
    Anthropic we convert to the native tool format and mark the last tool
    with ``cache_control`` so the full tools array is cached across turns.
    """
    if not hasattr(chat_model, "bind_tools"):
        return chat_model
    if not anthropic or not tool_schemas:
        return chat_model.bind_tools(tool_schemas)
    anthropic_tools: list[dict[str, Any]] = []
    for schema in tool_schemas:
        fn = schema.get("function", {}) if schema.get("type") == "function" else schema
        anthropic_tools.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or fn.get("input_schema") or {},
            }
        )
    anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
    return chat_model.bind_tools(anthropic_tools)


def _find_turn_user_index(messages: list[BaseMessage]) -> int:
    """Index of this turn's user query — the last non-nudge ``HumanMessage``.

    Nudges (``CONTINUE_NUDGE`` / ``EMPTY_NUDGE`` / ``AFTER_TOOL_NUDGE``) are
    injected by ``progress_continue`` and should NOT be treated as the turn
    boundary — the real user query sits further back. Falls back to 0 when
    no user-authored human message exists (empty list, or only nudges).
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if getattr(msg, "type", "") != "human":
            continue
        if extract_text(msg).strip() in _NUDGE_TEXTS:
            continue
        return i
    return 0


def _with_cache_control(message: BaseMessage) -> BaseMessage:
    """Return a deep copy of ``message`` with a ``cache_control`` marker on
    its last content block.

    Three content shapes to handle:
      * ``str`` → wrap into a single ``text`` block carrying cache_control.
      * ``list[dict]`` → clone the last dict and add cache_control to the
        clone so the caller's original content stays untouched.
      * anything else (empty / None / list of non-dicts) → append a
        dedicated empty-text block to carry the marker.

    The caller owns the copy and is free to re-assign it back into the
    per-call prompt list without mutating the persisted state.
    """
    copy = deepcopy(message)
    content = copy.content
    if isinstance(content, str):
        copy.content = [
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
        ]
        return copy
    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            new_last = dict(last)
            new_last["cache_control"] = {"type": "ephemeral"}
            content[-1] = new_last
        else:
            content.append(
                {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}}
            )
        return copy
    copy.content = [
        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}}
    ]
    return copy


def _build_prompt_messages(
    state_messages: list[BaseMessage],
    *,
    static_text: str,
    dynamic_text: str,
    anthropic: bool,
) -> list[BaseMessage]:
    """Assemble the full prompt for one model call.

    Layout (same for every provider; Anthropic additionally drops
    ``cache_control`` markers on two message positions):

        [System(static_text)]
        [...prior history]                       ← cc on tail (Anthropic)
        [Pseudo-HumanMessage(dynamic_text)]      ← only if dynamic non-empty
        [current user query]
        [...within-turn AI / tool messages]      ← cc on very last (Anthropic)

    Dynamic context lives PAST the history breakpoint, so it never poisons
    the cross-turn cache. The pseudo message is NOT persisted into
    ``state.messages`` — it exists only inside this per-call prompt list,
    so next turn's history stays byte-identical to what was cached here.
    """
    prompt: list[BaseMessage] = []
    if static_text:
        if anthropic:
            prompt.append(
                SystemMessage(
                    content=[
                        {
                            "type": "text",
                            "text": static_text,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                )
            )
        else:
            prompt.append(SystemMessage(content=static_text))

    messages = list(state_messages)
    if not messages and not dynamic_text:
        return prompt

    turn_idx = _find_turn_user_index(messages)
    if dynamic_text:
        pseudo = HumanMessage(
            content=f"<session-context>\n{dynamic_text}\n</session-context>"
        )
        # Insert BEFORE the user query. Indices of prior-turn content
        # (0..turn_idx-1) are unchanged by this insert; only turn_idx
        # onward shifts by one.
        messages.insert(turn_idx, pseudo)

    if anthropic:
        prior_hist_end = turn_idx - 1
        if 0 <= prior_hist_end < len(messages):
            messages[prior_hist_end] = _with_cache_control(messages[prior_hist_end])
        if messages:
            # Avoid double-marking when the prior-hist tail IS the request
            # tail (degenerate: nothing after the user query, no pseudo).
            if prior_hist_end != len(messages) - 1:
                messages[-1] = _with_cache_control(messages[-1])

    prompt.extend(messages)
    return prompt


def _build_graph(*, services: Any, chat_model: Any, event_sink) -> StateGraph:
    tool_adapter = ToolGraphAdapter(services.tools, services.permissions)
    is_anthropic = getattr(services.settings, "kind", "") == "anthropic"
    bound_model = _bind_tools_with_caching(
        chat_model, services.tools.tool_schemas(), anthropic=is_anthropic
    )

    def emit(kind: str, payload: dict[str, Any]) -> None:
        writer = get_stream_writer()
        writer({"kind": kind, "payload": payload})

    def prompt_assembly(state: GraphState) -> GraphState:
        user_query = find_latest_user_query(state.get("messages", []))
        static_text, dynamic_text = assemble_system_prompt_split(
            base_prompt=services.settings.system_prompt,
            services=services,
            mode=state.get("mode", "default"),
            latest_user_query=user_query,
        )
        # `prompt_context` keeps the concatenated form for any path that
        # doesn't care about the split (e.g. OpenAI auto-caching). The
        # split fields are consumed by model_call when targeting Anthropic
        # so the static prefix can carry cache_control.
        prompt_context = (
            f"{static_text}\n\n{dynamic_text}" if dynamic_text else static_text
        )
        return {
            "prompt_context": prompt_context,
            "prompt_context_static": static_text,
            "prompt_context_dynamic": dynamic_text,
            "user_query": user_query,
        }

    def maybe_compact(state: GraphState) -> GraphState:
        context_service = getattr(services, "context", None)
        if context_service is None:
            return {}
        messages = list(state.get("messages", []))
        if not context_service.should_compact(messages):
            return {}
        compacted, result = context_service.compact(messages)
        emit(
            "context_compacted",
            {
                "removed": result.removed_count,
                "before_tokens": result.before_tokens,
                "after_tokens": result.after_tokens,
            },
        )
        return {"messages": compacted}

    def model_call(state: GraphState) -> GraphState:
        prompt_messages = _build_prompt_messages(
            list(state.get("messages", [])),
            static_text=state.get("prompt_context_static", "")
            or state.get("prompt_context", ""),
            dynamic_text=state.get("prompt_context_dynamic", ""),
            anthropic=is_anthropic,
        )

        def _stream_collect() -> AIMessage:
            """Stream tokens as they arrive, accumulate into a full AIMessage."""
            emit("model_call_start", {})
            accumulated: AIMessage | None = None
            for chunk in bound_model.stream(prompt_messages):
                content = getattr(chunk, "content", "")
                delta_text = content if isinstance(content, str) else extract_text(chunk)
                if delta_text:
                    emit("assistant_token", {"text": delta_text})
                thinking_delta = extract_thinking_delta(chunk)
                if thinking_delta:
                    emit("assistant_thinking", {"text": thinking_delta})
                accumulated = chunk if accumulated is None else accumulated + chunk
            return accumulated if accumulated is not None else AIMessage(content="")

        t0 = time.time()
        try:
            ai_message = retry_with_backoff(_stream_collect)
        except APIRetryExhaustedError as exc:
            _fire_hook(services, "on_error", error=str(exc))
            cost = getattr(services, "cost", None)
            if cost is not None:
                cost.record_error(services.settings.model)
            return {
                "final_text": f"Model retry budget exhausted: {exc}",
                "error": f"retry_exhausted:{exc}",
                "route": "memory_update",
            }
        duration_ms = int((time.time() - t0) * 1000)

        if not isinstance(ai_message, AIMessage):
            ai_message = AIMessage(content=str(ai_message))

        cost = getattr(services, "cost", None)
        if cost is not None:
            usage = getattr(ai_message, "usage_metadata", None) or {}
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            # LangChain surfaces provider-native cache stats under
            # input_token_details (Anthropic: cache_read / cache_creation;
            # OpenAI: cached_tokens under `cache_read`). Capture both so
            # /cost can show a real hit rate instead of pretending the
            # whole prefix was uncached.
            details = usage.get("input_token_details") or {}
            cache_read = int(details.get("cache_read", 0) or 0)
            cache_write = int(
                details.get("cache_creation", 0) or details.get("cache_write", 0) or 0
            )
            if input_tokens or output_tokens or cache_read or cache_write:
                if cache_read or cache_write:
                    emit(
                        "cache_usage",
                        {"read": cache_read, "write": cache_write},
                    )
                cost.record_api_call(
                    services.settings.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    cache_read_tokens=cache_read,
                    cache_write_tokens=cache_write,
                )

        # Tokens were already emitted as they streamed; no post-hoc chunking.
        return {
            "messages": [ai_message],
            "step_count": state.get("step_count", 0) + 1,
        }

    def classify_output(state: GraphState) -> GraphState:
        if state.get("error"):
            return {
                "final_text": state.get("final_text", "Unknown error."),
                "error": state.get("error"),
                "route": "memory_update",
            }
        messages = list(state.get("messages", []))
        last_ai = next(
            (msg for msg in reversed(messages) if getattr(msg, "type", "") == "ai"), None
        )
        if last_ai is None:
            return {
                "final_text": "Model did not return an assistant message.",
                "error": "missing_ai",
                "route": "memory_update",
            }
        text, kind = marker_kind(extract_text(last_ai))
        tool_calls = list(getattr(last_ai, "tool_calls", []) or [])
        if state.get("step_count", 0) >= state.get("max_steps", 40):
            return {
                "final_text": "Reached the maximum step limit for this turn.",
                "error": "max_steps",
                "route": "memory_update",
            }
        if tool_calls:
            return {"route": "execute_tools"}
        if not text.strip():
            retries = state.get("empty_response_retries", 0)
            if retries < 1:
                return {"empty_response_retries": retries + 1, "route": "progress_continue"}
            return {
                "final_text": "Model returned an empty response twice.",
                "error": "empty_response",
                "route": "memory_update",
            }
        if kind == "progress":
            emit("progress", {"text": text})
            return {"route": "progress_continue"}
        emit("assistant_message", {"text": text})
        return {"final_text": text, "route": "memory_update"}

    def progress_continue(state: GraphState) -> GraphState:
        if state.get("empty_response_retries", 0) > 0:
            nudge = EMPTY_NUDGE
        elif state.get("saw_tool_result"):
            nudge = AFTER_TOOL_NUDGE
        else:
            nudge = CONTINUE_NUDGE
        return {"messages": [make_human_message(nudge)], "route": "model_call"}

    def execute_tools(state: GraphState) -> GraphState:
        # Tools run in a ThreadPoolExecutor (see graph_adapter.py). Those
        # worker threads don't inherit LangGraph's runnable context, so
        # calling get_stream_writer() from them — or any callable it
        # returned, since the writer re-reads get_config() on each call —
        # raises "Called get_config outside of a runnable context".
        # Route tool events through the user-supplied event_sink instead,
        # which is a plain callable with no context-var dependency.
        def thread_safe_emit(kind: str, payload: dict[str, Any]) -> None:
            if event_sink is not None:
                event_sink(kind, payload)

        last_ai = next(
            (
                msg
                for msg in reversed(state.get("messages", []))
                if getattr(msg, "type", "") == "ai"
            ),
            None,
        )
        if last_ai is None:
            return {
                "final_text": "Tool execution requested without an AI tool call.",
                "error": "tool_call_missing",
                "route": "memory_update",
            }

        tool_calls = list(getattr(last_ai, "tool_calls", []) or [])
        for call in tool_calls:
            name = (
                call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
            )
            args = (
                call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
            )
            _fire_hook(
                services,
                "pre_tool",
                tool_name=name,
                tool_input=args,
                thread_id=state.get("thread_id", ""),
            )

        batch = tool_adapter.execute(
            tool_calls,
            ToolContext(
                thread_id=state["thread_id"],
                cwd=services.settings.workspace,
                mode=state.get("mode", "default"),
                services=services,
                emit_event=thread_safe_emit,
                agent_name=getattr(services, "agent_name", "orchestrator"),
            ),
        )

        for tool_message in batch.tool_messages:
            _fire_hook(
                services,
                "post_tool",
                tool_name=getattr(tool_message, "name", "") or "",
                tool_output=extract_text(tool_message)[:500],
                is_error=getattr(tool_message, "status", "") == "error",
                thread_id=state.get("thread_id", ""),
            )

        updates: GraphState = {
            "messages": batch.tool_messages,
            "saw_tool_result": batch.saw_tool_result,
            "tool_error_count": state.get("tool_error_count", 0) + batch.tool_error_count,
        }
        if batch.await_user:
            updates["final_text"] = batch.final_text
            updates["await_user"] = True
            updates["route"] = "memory_update"
        else:
            updates["route"] = "model_call"
        return updates

    def memory_update(state: GraphState) -> GraphState:
        if state.get("final_text") and state.get("user_query"):
            services.memory.remember_turn(state["user_query"], state["final_text"])
        return {"route": "session_finalize"}

    def session_finalize(state: GraphState) -> GraphState:
        emit(
            "session_finalized",
            {
                "thread_id": state.get("thread_id", ""),
                "final_text": state.get("final_text"),
                "await_user": state.get("await_user", False),
            },
        )
        return state

    def route_from_state(state: GraphState) -> str:
        return state.get("route", "session_finalize")

    graph = StateGraph(GraphState)
    graph.add_node("prompt_assembly", prompt_assembly)
    graph.add_node("maybe_compact", maybe_compact)
    graph.add_node("model_call", model_call)
    graph.add_node("classify_output", classify_output)
    graph.add_node("progress_continue", progress_continue)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("memory_update", memory_update)
    graph.add_node("session_finalize", session_finalize)
    graph.add_edge(START, "prompt_assembly")
    graph.add_edge("prompt_assembly", "maybe_compact")
    graph.add_edge("maybe_compact", "model_call")
    graph.add_edge("model_call", "classify_output")
    graph.add_conditional_edges(
        "classify_output",
        route_from_state,
        {
            "execute_tools": "execute_tools",
            "progress_continue": "progress_continue",
            "memory_update": "memory_update",
            "session_finalize": "session_finalize",
        },
    )
    graph.add_conditional_edges(
        "execute_tools",
        route_from_state,
        {"model_call": "model_call", "memory_update": "memory_update"},
    )
    graph.add_edge("progress_continue", "model_call")
    graph.add_edge("memory_update", "session_finalize")
    graph.add_edge("session_finalize", END)
    return graph
