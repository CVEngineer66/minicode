from __future__ import annotations

from langgraph.types import interrupt


def ensure_tool_allowed(permissions: object, request: object, decision_key: str) -> dict[str, str]:
    """Pull a cached permission decision or surface an interrupt to the user.

    Recognized cached decisions:
    - allow_always / allow_turn / allow_all_turn  → pass through
    - deny_always                                 → return deny without prompting
    Anything else prompts via langgraph interrupt and persists allow_always/deny_always.
    """
    cached = permissions.cached_decision(decision_key)
    if cached in {"allow_always", "allow_turn", "allow_all_turn"}:
        return {"decision": cached}
    if cached == "deny_always":
        return {"decision": "deny"}
    payload = {
        "kind": request.kind,
        "summary": request.summary,
        "details": list(request.details),
        "scope": request.scope,
        "choices": list(request.choices),
    }
    decision = interrupt(payload)
    normalized = dict(decision or {})
    choice = normalized.get("decision")
    if choice in {"allow_always", "deny_always", "allow_turn", "allow_all_turn"}:
        permissions.store_decision(decision_key, choice, {"scope": request.scope})
    elif choice == "deny":
        # Normalize to a single one-shot deny
        normalized["decision"] = "deny_once"
    return normalized
