"""Model catalog — the single source of truth for "which models can I use".

Populated from ``config.json`` ``providers`` section, consumed by the TUI
model picker, the settings resolver, and ``create_chat_model``. Keeps
pricing (``features/cost``) and catalog concerns separate.

Config shape::

    {
      "providers": {
        "openai": {
          "kind": "openai",
          "api_key": "sk-...",
          "base_url": null,
          "extra_headers": {},
          "extra_body": {},
          "models": ["gpt-4o-mini", "gpt-4o"]
        },
        "anthropic": {
          "kind": "anthropic",
          "api_key": "sk-ant-...",
          "models": ["claude-opus-4-7"]
        }
      }
    }

Iteration order matters: the *first* provider's *first* model is used as the
session default. Users who want a different default move that entry to the
top of their config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProviderEntry:
    """One row of the catalog — a single provider with its credentials and model list."""

    name: str
    kind: str  # "openai" | "anthropic" — drives the SDK branch in model_factory
    api_key: str | None
    base_url: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)


class ModelCatalog:
    """Ordered collection of provider entries.

    ``providers`` preserves config.json's insertion order (Python 3.7+ dict
    guarantee) so ``default_identifier()`` returns the first-configured
    ``provider:model`` pair.
    """

    def __init__(self, providers: list[ProviderEntry]) -> None:
        self._providers: dict[str, ProviderEntry] = {p.name: p for p in providers}
        self._order: list[str] = [p.name for p in providers]

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> "ModelCatalog":
        providers: list[ProviderEntry] = []
        for name, payload in (raw or {}).items():
            if not isinstance(payload, dict):
                continue
            kind = str(payload.get("kind", "openai")).lower()
            if kind not in ("openai", "anthropic"):
                kind = "openai"
            models = [str(m) for m in payload.get("models", []) if isinstance(m, str) and m]
            if not models:
                continue
            providers.append(
                ProviderEntry(
                    name=str(name),
                    kind=kind,
                    api_key=payload.get("api_key") or None,
                    base_url=payload.get("base_url") or None,
                    extra_headers=dict(payload.get("extra_headers") or {}),
                    extra_body=dict(payload.get("extra_body") or {}),
                    models=models,
                )
            )
        return cls(providers)

    def is_empty(self) -> bool:
        return not self._providers

    def providers(self) -> list[ProviderEntry]:
        return [self._providers[name] for name in self._order]

    def get(self, provider_name: str) -> ProviderEntry | None:
        return self._providers.get(provider_name)

    def all_identifiers(self) -> list[str]:
        """Every ``provider:model`` pair in config order."""
        return [f"{p.name}:{model}" for name in self._order for p in [self._providers[name]] for model in p.models]

    def default_identifier(self) -> str | None:
        """First provider's first model — the session-default model.

        Returns ``None`` when no providers are configured; the runtime should
        surface "no model configured" rather than fall back to a hard-coded
        default.
        """
        for name in self._order:
            provider = self._providers[name]
            if provider.models:
                return f"{name}:{provider.models[0]}"
        return None

    def resolve(self, identifier: str) -> tuple[ProviderEntry, str] | None:
        """Parse ``provider:model`` and return (entry, model) or None.

        Tolerates models that contain a single colon in their name by joining
        all segments after the first as the model id (e.g. ``openai:org/model``
        would parse as provider=openai, model=org/model).
        """
        if ":" not in identifier:
            return None
        name, _, model = identifier.partition(":")
        entry = self._providers.get(name)
        if entry is None or model not in entry.models:
            return None
        return entry, model
