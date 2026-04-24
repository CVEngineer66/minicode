"""Settings loader.

Reads ``~/.minicode/config.json`` and projects it into a flat ``Settings``
dataclass for the rest of the runtime. The heavy lifting — "which providers
and models are available" — is delegated to ``ModelCatalog`` (see
``features/models/catalog.py``); ``Settings`` only carries the *currently
active* credentials and config knobs.

Active-model resolution order:
1. ``current_model`` (``"provider:model"``) if present and it resolves to a
   catalog entry. Written by the TUI ``/model`` picker via
   ``save_current_model``.
2. First provider's first model (config insertion order).
3. Empty — ``Settings.provider`` / ``model`` are blank strings and
   ``create_chat_model`` refuses with a clear error.

Environment variables: the only one consulted at config-load time is
``MINICODE_HOME`` (read by ``paths.py`` before this module is imported, so
it must live there). Provider/model/mode/thinking knobs all come from
config.json — switch them via the TUI (``/model``, ``/mode``), the
``minicode config`` CLI, or by editing ``config.json`` directly.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minicode.features.models import ModelCatalog, ProviderEntry

from .paths import AppPaths


@dataclass(slots=True)
class Settings:
    """Resolved active-session settings.

    ``provider`` / ``model`` / ``api_key`` / ``base_url`` / ``extra_*`` always
    reflect the *currently chosen* provider entry. ``catalog`` carries the
    full list so the TUI model picker can show all options.
    """

    provider: str
    model: str
    kind: str  # "openai" | "anthropic", drives create_chat_model branch
    base_url: str | None
    api_key: str | None
    auto_mode: str
    system_prompt: str
    workspace: str
    extra_headers: dict[str, str]
    extra_body: dict[str, Any]
    thinking: str = "auto"
    thinking_budget_tokens: int = 2048
    catalog: ModelCatalog = field(default_factory=lambda: ModelCatalog([]))

    def apply_provider(self, entry: ProviderEntry, model: str) -> None:
        """Mutate in place to point at a different catalog entry.

        Used by TUI ``/model``. Session-scoped unless the caller also writes
        ``save_current_model`` to persist the choice across restarts.
        """
        self.provider = entry.name
        self.model = model
        self.kind = entry.kind
        self.api_key = entry.api_key
        self.base_url = entry.base_url
        self.extra_headers = dict(entry.extra_headers)
        self.extra_body = dict(entry.extra_body)


def _load_config_data(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _pick_active(
    catalog: ModelCatalog, current_model: str | None
) -> tuple[ProviderEntry, str] | None:
    """Resolve the active (entry, model) per the documented order.

    ``current_model`` (from ``config.json``) wins when it parses and points
    to an entry that still exists in the catalog; stale values fall through
    to first-in-config. Returns ``None`` only when the catalog itself is
    empty.
    """
    if current_model:
        resolved = catalog.resolve(current_model)
        if resolved is not None:
            return resolved
    default_id = catalog.default_identifier()
    if default_id is None:
        return None
    return catalog.resolve(default_id)


def normalize_mode(value: Any) -> str:
    normalized = str(value or "default").strip().lower() or "default"
    if normalized in {"auto", "plan"}:
        return "default"
    if normalized not in {"default", "bypass"}:
        return "default"
    return normalized


def load_settings(paths: AppPaths, cwd: str | Path) -> Settings:
    config_data = _load_config_data(paths.config_path)

    catalog = ModelCatalog.from_config(config_data.get("providers") or {})
    current_model = config_data.get("current_model")
    active = _pick_active(catalog, current_model if isinstance(current_model, str) else None)

    auto_mode = normalize_mode(config_data.get("default_mode", "default"))
    system_prompt = str(
        config_data.get(
            "system_prompt",
            "You are MiniCode, a coding assistant focused on safe, concrete execution.",
        )
    )

    thinking_raw = config_data.get("thinking", "auto")
    thinking = str(thinking_raw).strip().lower() or "auto"
    if thinking not in ("auto", "on", "off"):
        thinking = "auto"
    try:
        thinking_budget_tokens = max(0, int(config_data.get("thinking_budget_tokens", 2048)))
    except (TypeError, ValueError):
        thinking_budget_tokens = 2048

    if active is None:
        return Settings(
            provider="",
            model="",
            kind="openai",
            base_url=None,
            api_key=None,
            auto_mode=auto_mode,
            system_prompt=system_prompt,
            workspace=str(Path(cwd).resolve()),
            extra_headers={},
            extra_body={},
            thinking=thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            catalog=catalog,
        )

    entry, model = active
    return Settings(
        provider=entry.name,
        model=model,
        kind=entry.kind,
        base_url=entry.base_url,
        api_key=entry.api_key,
        auto_mode=auto_mode,
        system_prompt=system_prompt,
        workspace=str(Path(cwd).resolve()),
        extra_headers=dict(entry.extra_headers),
        extra_body=dict(entry.extra_body),
        thinking=thinking,
        thinking_budget_tokens=thinking_budget_tokens,
        catalog=catalog,
    )


# ---------------------------------------------------------------------------
# Persistent config mutation helpers — used by the TUI model picker. All
# writes go through ``_atomic_write_json`` so a partial write can't leave
# the user with a corrupt config.
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_current_model(paths: AppPaths, identifier: str) -> None:
    """Persist the TUI model picker selection to ``config.json``.

    Writes the ``"current_model"`` field (format: ``"provider:model"``).
    Missing config files are created with just ``providers: {}`` so the user
    has somewhere to put the key.
    """
    data = _load_config_data(paths.config_path)
    if "providers" not in data:
        data["providers"] = {}
    data["current_model"] = identifier
    _atomic_write_json(paths.config_path, data)


# ---------------------------------------------------------------------------
# First-run scaffolding — when the user has nothing yet, lay down:
#   * an empty-but-valid ``config.json`` so every reader sees a parseable file
#   * a ``config.example.json`` next to it with commented examples for each
#     supported provider shape so users know what to copy/adapt
# Neither file is overwritten if it already exists.
# ---------------------------------------------------------------------------


_EXAMPLE_CONFIG = """{
  "_README": "Copy the blocks you want into ~/.minicode/config.json under `providers`, fill in `api_key`, then restart MiniCode. `current_model` is set automatically when you pick a model in the TUI via /model.",

  "providers": {
    "openai": {
      "kind": "openai",
      "api_key": "sk-REPLACE-ME",
      "models": ["gpt-4o-mini", "gpt-4o"]
    },
    "anthropic": {
      "kind": "anthropic",
      "api_key": "sk-ant-REPLACE-ME",
      "models": ["claude-opus-4-7", "claude-sonnet-4-6"]
    },
    "dashscope": {
      "kind": "openai",
      "api_key": "sk-REPLACE-ME",
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "extra_body": { "enable_thinking": true },
      "models": ["qwen3-30b-a3b", "qwen-max-latest"]
    },
    "openrouter": {
      "kind": "openai",
      "api_key": "sk-or-REPLACE-ME",
      "base_url": "https://openrouter.ai/api/v1",
      "extra_headers": { "HTTP-Referer": "https://example.com", "X-Title": "MiniCode" },
      "models": ["anthropic/claude-sonnet-4", "openai/gpt-4o"]
    }
  },

  "current_model": "openai:gpt-4o-mini",
  "default_mode": "default",
  "system_prompt": "You are MiniCode, a coding assistant focused on safe, concrete execution.",
  "thinking": "auto",
  "thinking_budget_tokens": 2048
}
"""


def ensure_config_scaffold(paths: AppPaths) -> None:
    """On first launch, lay down an empty ``config.json`` + an example file.

    - ``config.json``: valid empty config (``{"providers": {}}``). Created
      only when absent — never touched afterwards.
    - ``config.example.json``: commented template in ``~/.minicode/``.
      Created when absent; not overwritten if the user edited it.

    Safe to call on every startup; both writes are no-ops once the files
    exist.
    """
    if not paths.config_path.exists():
        try:
            _atomic_write_json(paths.config_path, {"providers": {}})
        except OSError:
            pass
    example_path = paths.config_path.with_name("config.example.json")
    if not example_path.exists():
        try:
            example_path.parent.mkdir(parents=True, exist_ok=True)
            example_path.write_text(_EXAMPLE_CONFIG, encoding="utf-8")
        except OSError:
            pass
