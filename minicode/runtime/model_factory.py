from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from minicode.platform.config import Settings


@dataclass(slots=True)
class ModelInfo:
    provider: str
    model: str
    supports_tools: bool = True


def resolve_model_info(settings: Settings) -> ModelInfo:
    unsupported = {"gpt-3.5-turbo-instruct", "text-davinci-003"}
    return ModelInfo(
        provider=settings.provider,
        model=settings.model,
        supports_tools=settings.model not in unsupported,
    )


# ---------------------------------------------------------------------------
# Thinking / reasoning resolution
# ---------------------------------------------------------------------------
#
# Three-layer policy:
#   1. settings.thinking = "on" | "off" -> absolute.
#   2. settings.thinking = "auto"       -> mode decides:
#        bypass  -> off  (user wants speed / low token spend)
#        other   -> on   (hybrid-thinking models self-regulate depth)
#   3. Provider family maps the boolean to the concrete knob each API expects.
#
# Known families (model-name prefix, case-insensitive):
#   anthropic + claude-(opus|sonnet|haiku)-(4|5)-  -> thinking={"type":..., "budget_tokens":N}
#   qwen / qwq / tongyi                             -> extra_body.enable_thinking
#   glm / chatglm                                   -> extra_body.thinking={"type": enabled|disabled}
#   o1 / o3 / o4 / gpt-5                            -> reasoning_effort kwarg (medium when on, minimal when off)
#   deepseek-reasoner / deepseek-chat               -> no-op (model name already determines)
#   anything else                                   -> no-op


def _model_family(settings: Settings) -> str:
    if settings.kind == "anthropic":
        return "anthropic"
    name = (settings.model or "").lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("qwen", "qwq", "tongyi")):
        return "qwen"
    if name.startswith(("glm", "chatglm")):
        return "zhipu"
    if name.startswith(("o1", "o3", "o4", "gpt-5")):
        return "openai_reasoning"
    return "none"


def resolve_thinking_enabled(settings: Settings, mode: str | None) -> bool:
    policy = (settings.thinking or "auto").lower()
    if policy == "on":
        return True
    if policy == "off":
        return False
    # auto: let the mode steer.
    if mode == "bypass":
        return False
    # Default mode: let the hybrid-thinking models self-regulate.
    return True


def _apply_anthropic_thinking(kwargs: dict[str, Any], enabled: bool, budget: int) -> None:
    # Only Claude 4.x and later accept the `thinking` parameter. Older models
    # reject it with a 400, so we gate on the name prefix.
    model = str(kwargs.get("model", "")).lower()
    supports = any(
        model.startswith(f"claude-{tier}-{gen}")
        for tier in ("opus", "sonnet", "haiku")
        for gen in ("4", "5")
    )
    if not supports:
        return
    if not enabled:
        kwargs["thinking"] = {"type": "disabled"}
        return
    # Opus 4.7 rejects `{type: "enabled", budget_tokens}` with a 400 — adaptive
    # is the only legal on-mode. 4.6 (Opus + Sonnet) deprecated budget_tokens
    # and also prefers adaptive. Older models (4.5 and below) keep the old
    # shape. Claude decides its own thinking depth under adaptive; no budget.
    uses_adaptive = any(
        model.startswith(f"claude-{tier}-{gen}")
        for tier in ("opus", "sonnet", "haiku")
        for gen in ("4-6", "4-7", "5")
    )
    if uses_adaptive:
        kwargs["thinking"] = {"type": "adaptive"}
    elif budget > 0:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
    else:
        kwargs["thinking"] = {"type": "disabled"}


def _apply_qwen_thinking(kwargs: dict[str, Any], enabled: bool, budget: int) -> None:
    _ = budget
    extra = dict(kwargs.get("extra_body") or {})
    extra["enable_thinking"] = bool(enabled)
    kwargs["extra_body"] = extra


def _apply_zhipu_thinking(kwargs: dict[str, Any], enabled: bool, budget: int) -> None:
    _ = budget
    extra = dict(kwargs.get("extra_body") or {})
    extra["thinking"] = {"type": "enabled" if enabled else "disabled"}
    kwargs["extra_body"] = extra


def _apply_openai_reasoning(kwargs: dict[str, Any], enabled: bool, budget: int) -> None:
    # For GPT-5 / o-series we want reasoning summaries to surface back into the
    # AIMessage content stream; ``output_version="responses/v1"`` opts LangChain
    # into the Responses API shape where ``content`` can contain ``type:
    # "reasoning"`` blocks with ``summary[].text``.
    _ = budget
    kwargs["output_version"] = "responses/v1"
    kwargs["reasoning"] = {
        "effort": "medium" if enabled else "none",
        "summary": "auto" if enabled else None,
    }


_ThinkingApplier = Callable[[dict[str, Any], bool, int], None]
_THINKING_APPLIERS: dict[str, _ThinkingApplier] = {
    "anthropic": _apply_anthropic_thinking,
    "qwen": _apply_qwen_thinking,
    "zhipu": _apply_zhipu_thinking,
    "openai_reasoning": _apply_openai_reasoning,
}


def apply_thinking_to_kwargs(
    settings: Settings,
    kwargs: dict[str, Any],
    *,
    enabled: bool,
    budget_tokens: int,
) -> None:
    """Provider-aware thinking knob. Mutates kwargs in place.

    Families not in ``_THINKING_APPLIERS`` (``none`` / deepseek / etc.) are
    no-ops — either the model name alone determines reasoning, or the
    provider doesn't expose a thinking toggle at all.
    """
    applier = _THINKING_APPLIERS.get(_model_family(settings))
    if applier is not None:
        applier(kwargs, enabled, budget_tokens)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _import_chat_class(kind: str) -> type:
    try:
        if kind == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic
        from langchain_openai import ChatOpenAI

        return ChatOpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing LangGraph dependencies. Install the langgraph optional dependencies."
        ) from exc


def create_chat_model(settings: Settings, mode: str | None = None) -> Any:
    if not settings.provider or not settings.model:
        raise RuntimeError(
            "No model configured. Add a `providers` block to "
            "~/.minicode/config.json (see README.md) or use /model in the TUI "
            "once a provider exists."
        )
    info = resolve_model_info(settings)
    if not info.supports_tools:
        raise RuntimeError(
            f"Model {info.model} does not support tool calling. Choose a tool-capable model."
        )
    if not settings.api_key:
        raise RuntimeError(
            f"Provider `{settings.provider}` has no api_key set in config.json."
        )

    chat_cls = _import_chat_class(settings.kind)
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "api_key": settings.api_key,
        # Streaming request default per Anthropic docs — leaves headroom for
        # Opus 4.6/4.7's 128K output ceiling without risking mid-generation
        # cutoffs on longer answers. Callers that want more can bump it.
        "max_tokens": 16000,
    }
    # OpenAI-compatible endpoints accept base_url / default_headers / extra_body;
    # ChatAnthropic does not — keep the extras kind-gated.
    if settings.kind == "anthropic":
        kwargs["default_headers"] = {
            "anthropic-beta": "context-1m-2025-08-07"
        }
    if settings.kind != "anthropic":
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        if settings.extra_headers:
            kwargs["default_headers"] = settings.extra_headers
        if settings.extra_body:
            kwargs["extra_body"] = dict(settings.extra_body)

    apply_thinking_to_kwargs(
        settings,
        kwargs,
        enabled=resolve_thinking_enabled(settings, mode),
        budget_tokens=int(getattr(settings, "thinking_budget_tokens", 0) or 0),
    )
    return chat_cls(**kwargs)
