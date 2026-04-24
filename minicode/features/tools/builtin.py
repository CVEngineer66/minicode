"""Builtin tools for MiniCode's runtime.

Every tool is declared with the ``@tools.register`` decorator from
``decorator.py``. The decorator infers the JSON Schema from the function
signature and docstring and collects a ``ToolSpec`` into the module-level
``tools`` registrar; ``build_builtin_registry`` hands that list to the
``ToolRegistry`` for the adapter to execute.

Parameter conventions:
- Positional/keyword params with a default are optional in the schema.
- Params without a default are required.
- ``*, context: ToolContext`` is injected by the adapter — it does NOT appear
  in the schema.
- Plain Python types (``str``/``int``/``bool``/``float``/``list``/``dict``)
  map to the matching JSON Schema types. ``Any`` / unannotated / ``Optional``
  are permissive.
"""

from __future__ import annotations

import ast
import base64
import csv
import difflib
import gzip
import hashlib
import hmac as hmac_lib
import io
import json
import random
import re
import shutil
import string
import tarfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from langgraph.types import interrupt

from minicode.core.types import PermissionPolicy, ToolCapability, ToolResult, ToolSpec
from minicode.features.tasks.services import normalize_task_tracker_item
from minicode.platform.http import http_request as _http_request_impl, simple_web_search
from minicode.platform.process import run_command_sync

from .decorator import ToolRegistrar, _passthrough_validator
from .metadata import enrich_input_schema, enrich_tool_description
from .registry import ToolRegistry
from .types import ToolContext


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _resolve_path(context: ToolContext, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = Path(context.cwd) / candidate
    return candidate.resolve()


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _normalize_choice_options(options: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, raw in enumerate(options):
        option_id = f"option_{index + 1}"
        label = ""
        description = ""
        if isinstance(raw, dict):
            option_id = str(
                raw.get("id")
                or raw.get("value")
                or raw.get("key")
                or option_id
            ).strip() or option_id
            label = str(
                raw.get("label")
                or raw.get("title")
                or raw.get("name")
                or option_id
            ).strip()
            description = str(
                raw.get("description")
                or raw.get("detail")
                or raw.get("details")
                or raw.get("tradeoff")
                or ""
            ).strip()
        else:
            label = str(raw).strip()
        if not label:
            label = option_id
        normalized.append(
            {
                "id": option_id,
                "label": label,
                "description": description,
            }
        )
    return normalized


def _normalize_write_scope(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        items = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    return []


def _build_worker_interrupt_payload(worker: Any, payload: dict[str, Any]) -> dict[str, Any]:
    interrupt_payload = dict(payload or {})
    details = [str(line) for line in interrupt_payload.get("details", [])]
    details.insert(0, f"Subagent: {worker.agent_name} [{worker.task_type}]")
    interrupt_payload["details"] = details
    interrupt_payload["summary"] = interrupt_payload.get("summary") or worker.title
    interrupt_payload["worker_id"] = worker.worker_id
    interrupt_payload["worker_thread_id"] = worker.thread_id
    choices = []
    for choice in list(interrupt_payload.get("choices", [])):
        updated = dict(choice or {})
        choice_payload = dict(
            updated.get("payload")
            or (
                {"decision": str(updated.get("decision"))}
                if updated.get("decision") is not None
                else {}
            )
        )
        choice_payload.setdefault("worker_id", worker.worker_id)
        updated["payload"] = choice_payload
        choices.append(updated)
    if choices:
        interrupt_payload["choices"] = choices
    cancel_payload = dict(interrupt_payload.get("cancel_payload") or {})
    cancel_payload.setdefault("worker_id", worker.worker_id)
    if not cancel_payload:
        cancel_payload = {"worker_id": worker.worker_id}
    interrupt_payload["cancel_payload"] = cancel_payload
    return interrupt_payload


def _run_worker_until_complete(worker_id: str, *, context: ToolContext) -> Any:
    collaboration = context.services.collaboration
    resume: dict[str, Any] | None = None
    while True:
        result = collaboration.start_worker(
            services=context.services,
            worker_id=worker_id,
            mode=context.mode,
            resume=resume,
        )
        interrupt_payload = getattr(result, "interrupt", None)
        if not interrupt_payload:
            return result
        worker = collaboration.get_worker(worker_id)
        decision = interrupt(_build_worker_interrupt_payload(worker, interrupt_payload))
        resume = dict(decision or {})


# ---------------------------------------------------------------------------
# Permission policy shorthands
# ---------------------------------------------------------------------------


def _always(kind: str) -> PermissionPolicy:
    return PermissionPolicy(kind=kind, always_require_approval=True)


# ---------------------------------------------------------------------------
# Tool registrar
# ---------------------------------------------------------------------------


tools = ToolRegistrar()


# --- user interaction ------------------------------------------------------


@tools.register(
    capability=ToolCapability(concurrency_safe=False, interactive=True),
    permission_policy=PermissionPolicy(kind="ask_user"),
)
def ask_user(question: str) -> ToolResult:
    """Ask the user a direct follow-up question and end the current turn."""
    return ToolResult(content=question, await_user=True, metadata={"await_user": True})


@tools.register(
    capability=ToolCapability(concurrency_safe=False, interactive=True),
    permission_policy=PermissionPolicy(kind="ask_user_choice"),
)
def ask_user_choice(question: str, options: list) -> ToolResult:
    """Present bounded options to the user and resume the same turn with their selection."""
    normalized = _normalize_choice_options(list(options or []))
    if not question.strip():
        return ToolResult(ok=False, content="Question cannot be empty.", error="empty_question")
    if len(normalized) < 2:
        return ToolResult(
            ok=False,
            content="ask_user_choice requires at least two options.",
            error="insufficient_options",
        )
    details: list[str] = []
    choices: list[dict[str, Any]] = []
    for index, option in enumerate(normalized, start=1):
        line = f"{index}. {option['label']}"
        if option["description"]:
            line += f" - {option['description']}"
        details.append(line)
        choices.append(
            {
                "key": str(index),
                "label": option["label"],
                "payload": {
                    "choice_id": option["id"],
                    "choice_label": option["label"],
                },
            }
        )
    selection = interrupt(
        {
            "prompt_kind": "choice",
            "summary": question,
            "details": details,
            "choices": choices,
            "cancel_payload": {"choice_cancelled": True},
        }
    )
    if not isinstance(selection, dict):
        return ToolResult(
            ok=False,
            content="User selection payload was invalid.",
            error="invalid_choice_payload",
        )
    if selection.get("choice_cancelled"):
        return ToolResult(
            ok=False,
            content="User cancelled option selection.",
            error="choice_cancelled",
        )
    selected_id = str(selection.get("choice_id", "")).strip()
    selected = next((item for item in normalized if item["id"] == selected_id), None)
    if selected is None:
        return ToolResult(
            ok=False,
            content="User selection did not match any offered option.",
            error="unknown_choice",
        )
    payload = {"selected": selected}
    return ToolResult(
        content=_json_dump(payload),
        structured=payload,
        metadata=payload,
    )


# --- filesystem read -------------------------------------------------------


@tools.register(capability=ToolCapability(reads_files=True))
def list_files(
    path: str = ".",
    recursive: bool = False,
    include_hidden: bool = False,
    *,
    context: ToolContext,
) -> ToolResult:
    """List files in a directory."""
    root = _resolve_path(context, path)
    items = []
    iterator = root.rglob("*") if recursive else root.iterdir()
    for item in iterator:
        rel_name = str(item.relative_to(root if root.is_dir() else root.parent))
        if not include_hidden and any(part.startswith(".") for part in item.parts):
            continue
        items.append({"path": rel_name, "is_dir": item.is_dir()})
    return ToolResult(content=_json_dump(items[:1000]))


@tools.register(capability=ToolCapability(reads_files=True))
def file_tree(path: str = ".", max_depth: int = 4, *, context: ToolContext) -> ToolResult:
    """Render a directory tree."""
    root = _resolve_path(context, path)
    lines: list[str] = [root.name]
    for child in sorted(root.rglob("*")):
        try:
            depth = len(child.relative_to(root).parts)
        except ValueError:
            continue
        if depth > max_depth:
            continue
        indent = "  " * depth
        lines.append(f"{indent}- {child.name}{'/' if child.is_dir() else ''}")
    return ToolResult(content="\n".join(lines))


@tools.register(capability=ToolCapability(reads_files=True))
def grep_files(
    pattern: str,
    path: str = ".",
    ignore_case: bool = False,
    max_matches: int = 100,
    *,
    context: ToolContext,
) -> ToolResult:
    """Search file contents using a regex pattern."""
    root = _resolve_path(context, path)
    regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    matches: list[dict[str, Any]] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        text = _read_text(file_path)
        for index, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append({"file": str(file_path.relative_to(root)), "line": index, "text": line})
                if len(matches) >= max_matches:
                    return ToolResult(content=_json_dump(matches))
    return ToolResult(content=_json_dump(matches))


@tools.register(capability=ToolCapability(reads_files=True))
def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    *,
    context: ToolContext,
) -> ToolResult:
    """Read a text file."""
    resolved = _resolve_path(context, path)
    text = _read_text(resolved)
    if start_line or end_line:
        lines = text.splitlines()
        start = max(int(start_line or 1), 1)
        end = min(int(end_line or len(lines)), len(lines))
        text = "\n".join(lines[start - 1 : end])
    return ToolResult(content=text)


# --- filesystem write ------------------------------------------------------


@tools.register(
    capability=ToolCapability(writes_files=True, concurrency_safe=False),
    permission_policy=_always("write_file"),
)
def write_file(path: str, content: str, append: bool = False, *, context: ToolContext) -> ToolResult:
    """Write text to a file."""
    resolved = _resolve_path(context, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(resolved, mode, encoding="utf-8") as handle:
        handle.write(content)
    return ToolResult(content=f"Wrote {resolved}")


@tools.register(
    capability=ToolCapability(writes_files=True, concurrency_safe=False),
    permission_policy=_always("modify_file"),
)
def modify_file(
    path: str,
    search: str,
    replace: str = "",
    count: int = 0,
    *,
    context: ToolContext,
) -> ToolResult:
    """Replace text inside a file."""
    resolved = _resolve_path(context, path)
    text = _read_text(resolved)
    updated = text.replace(search, replace, count) if count > 0 else text.replace(search, replace)
    resolved.write_text(updated, encoding="utf-8")
    return ToolResult(content=f"Updated {resolved}")


@tools.register(
    capability=ToolCapability(writes_files=True, concurrency_safe=False),
    permission_policy=_always("edit_file"),
)
def edit_file(path: str, search: str, replace: str, *, context: ToolContext) -> ToolResult:
    """Alias of modify_file for direct text edits."""
    return modify_file(path=path, search=search, replace=replace, context=context)


@tools.register(
    capability=ToolCapability(writes_files=True, concurrency_safe=False),
    permission_policy=_always("patch_file"),
)
def patch_file(path: str, updated_content: str = "", *, context: ToolContext) -> ToolResult:
    """Overwrite a file with updated content."""
    resolved = _resolve_path(context, path)
    resolved.write_text(updated_content, encoding="utf-8")
    return ToolResult(content=f"Patched {resolved}")


# --- batch file operations -------------------------------------------------


@tools.register(
    capability=ToolCapability(writes_files=True, concurrency_safe=False),
    permission_policy=_always("batch_copy"),
)
def batch_copy(items: list, *, context: ToolContext) -> ToolResult:
    """Copy multiple files or directories."""
    results = []
    for item in items:
        source = _resolve_path(context, item["source"])
        target = _resolve_path(context, item["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
        results.append({"source": str(source), "target": str(target)})
    return ToolResult(content=_json_dump(results))


@tools.register(
    capability=ToolCapability(writes_files=True, concurrency_safe=False),
    permission_policy=_always("batch_move"),
)
def batch_move(items: list, *, context: ToolContext) -> ToolResult:
    """Move multiple files or directories."""
    results = []
    for item in items:
        source = _resolve_path(context, item["source"])
        target = _resolve_path(context, item["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        results.append({"source": str(source), "target": str(target)})
    return ToolResult(content=_json_dump(results))


@tools.register(
    capability=ToolCapability(writes_files=True, concurrency_safe=False),
    permission_policy=_always("batch_delete"),
)
def batch_delete(paths: list, *, context: ToolContext) -> ToolResult:
    """Delete multiple files or directories."""
    deleted = []
    for item in paths:
        resolved = _resolve_path(context, item)
        if resolved.is_dir():
            shutil.rmtree(resolved)
        elif resolved.exists():
            resolved.unlink()
        deleted.append(str(resolved))
    return ToolResult(content=_json_dump(deleted))


# --- shell / process / network --------------------------------------------


@tools.register(
    capability=ToolCapability(shell=True, concurrency_safe=False, long_running=True),
    permission_policy=_always("run_command"),
)
def run_command(
    command: str,
    timeout: int = 60,
    background: bool = False,
    *,
    context: ToolContext,
) -> ToolResult:
    """Run a shell command."""
    if background:
        record = context.services.background_tasks.start(command, context.cwd)
        return ToolResult(
            content=_json_dump(
                {"task_id": record.task_id, "status": record.status, "output_path": record.output_path}
            )
        )
    code, stdout, stderr = run_command_sync(command, context.cwd, timeout)
    payload = {"return_code": code, "stdout": stdout, "stderr": stderr}
    return ToolResult(ok=code == 0, content=_json_dump(payload), error=None if code == 0 else stderr[:500])


@tools.register()
def background_tasks_list(limit: int = 20, *, context: ToolContext) -> ToolResult:
    """List tracked background shell tasks after refreshing their latest status."""
    records = context.services.background_tasks.refresh()
    payload = {
        "tasks": list(records[: max(1, int(limit or 20))]),
        "total": len(records),
        "slots": context.services.background_tasks.slot_stats(),
    }
    return ToolResult(content=_json_dump(payload), structured=payload, metadata=payload)


@tools.register()
def background_task_status(task_id: str, *, context: ToolContext) -> ToolResult:
    """Return the latest metadata for one background shell task by id."""
    record = context.services.background_tasks.get(task_id, refresh=True)
    if record is None:
        return ToolResult(
            ok=False,
            content=f"Background task `{task_id}` was not found.",
            error="background_task_not_found",
        )
    return ToolResult(content=_json_dump(record), structured=record, metadata=record)


@tools.register()
def background_task_output(task_id: str, *, context: ToolContext) -> ToolResult:
    """Read the latest captured output for one background shell task by id."""
    record = context.services.background_tasks.get(task_id, refresh=True)
    if record is None:
        return ToolResult(
            ok=False,
            content=f"Background task `{task_id}` was not found.",
            error="background_task_not_found",
        )
    payload = {
        "task": record,
        "output": context.services.background_tasks.read_output(task_id),
    }
    return ToolResult(content=_json_dump(payload), structured=payload, metadata=payload)


@tools.register(
    capability=ToolCapability(network=True),
    permission_policy=_always("web_fetch"),
)
def web_fetch(url: str, timeout: int = 30) -> ToolResult:
    """Fetch the contents of a URL."""
    status, content = _http_request_impl("GET", url, timeout=timeout)
    return ToolResult(ok=status < 400, content=content[:10000], error=None if status < 400 else f"HTTP {status}")


@tools.register(
    capability=ToolCapability(network=True),
    permission_policy=_always("web_search"),
)
def web_search(query: str) -> ToolResult:
    """Search the web."""
    return ToolResult(content=simple_web_search(query))


@tools.register(
    capability=ToolCapability(network=True),
    permission_policy=_always("http_request"),
)
def http_request(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    body: Any = None,
    timeout: int = 30,
) -> ToolResult:
    """Send an HTTP request."""
    status, content = _http_request_impl(
        method,
        url,
        headers=dict(headers or {}),
        body=body,
        timeout=timeout,
    )
    return ToolResult(ok=status < 400, content=_json_dump({"status": status, "body": content[:10000]}))


# --- text / data utilities -------------------------------------------------


@tools.register()
def json_format(text: str) -> ToolResult:
    """Pretty print JSON text."""
    return ToolResult(content=_json_dump(json.loads(text)))


@tools.register()
def json_parse(text: str) -> ToolResult:
    """Parse JSON text."""
    parsed = json.loads(text)
    return ToolResult(structured=parsed, content=_json_dump(parsed))


@tools.register()
def regex_test(pattern: str, text: str) -> ToolResult:
    """Run a regex against text."""
    regex = re.compile(pattern)
    matches = [{"match": item.group(0), "span": list(item.span())} for item in regex.finditer(text)]
    return ToolResult(content=_json_dump(matches))


@tools.register()
def regex_replace(pattern: str, text: str, replace: str = "") -> ToolResult:
    """Replace regex matches in text."""
    return ToolResult(content=re.sub(pattern, replace, text))


@tools.register()
def base64_encode(text: str) -> ToolResult:
    """Encode text as base64."""
    return ToolResult(content=base64.b64encode(text.encode("utf-8")).decode("ascii"))


@tools.register()
def base64_decode(text: str) -> ToolResult:
    """Decode base64 text."""
    return ToolResult(content=base64.b64decode(text.encode("ascii")).decode("utf-8", errors="replace"))


@tools.register()
def url_encode(text: str) -> ToolResult:
    """URL encode text."""
    return ToolResult(content=quote(text))


@tools.register()
def url_decode(text: str) -> ToolResult:
    """URL decode text."""
    return ToolResult(content=unquote(text))


@tools.register()
def current_time() -> ToolResult:
    """Get the current local time."""
    return ToolResult(content=time.strftime("%Y-%m-%d %H:%M:%S"))


@tools.register()
def timestamp() -> ToolResult:
    """Get the current unix timestamp."""
    return ToolResult(content=str(int(time.time())))


@tools.register()
def hash(text: str, algorithm: str = "sha256") -> ToolResult:
    """Hash text using a named algorithm."""
    hasher = hashlib.new(algorithm.lower())
    hasher.update(text.encode("utf-8"))
    return ToolResult(content=hasher.hexdigest())


@tools.register()
def hmac(text: str, key: str, algorithm: str = "sha256") -> ToolResult:
    """Create an HMAC digest."""
    algo = getattr(hashlib, algorithm.lower())
    digest = hmac_lib.new(key.encode("utf-8"), text.encode("utf-8"), algo)
    return ToolResult(content=digest.hexdigest())


# --- archives --------------------------------------------------------------


@tools.register(
    capability=ToolCapability(reads_files=True, writes_files=True, concurrency_safe=False),
    permission_policy=_always("archive"),
)
def gzip_compress(path: str, *, context: ToolContext) -> ToolResult:
    """Compress a file with gzip."""
    resolved = _resolve_path(context, path)
    target = resolved.with_suffix(resolved.suffix + ".gz")
    with open(resolved, "rb") as source, gzip.open(target, "wb") as destination:
        shutil.copyfileobj(source, destination)
    return ToolResult(content=str(target))


@tools.register(
    capability=ToolCapability(reads_files=True, writes_files=True, concurrency_safe=False),
    permission_policy=_always("archive"),
)
def gzip_decompress(path: str, *, context: ToolContext) -> ToolResult:
    """Decompress a gzip file."""
    resolved = _resolve_path(context, path)
    target = Path(str(resolved).removesuffix(".gz"))
    with gzip.open(resolved, "rb") as source, open(target, "wb") as destination:
        shutil.copyfileobj(source, destination)
    return ToolResult(content=str(target))


@tools.register(
    capability=ToolCapability(reads_files=True, writes_files=True, concurrency_safe=False),
    permission_policy=_always("archive"),
)
def tar_create(target: str, paths: list, *, context: ToolContext) -> ToolResult:
    """Create a tar archive."""
    destination = _resolve_path(context, target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w") as tar:
        for item in paths:
            resolved = _resolve_path(context, item)
            tar.add(resolved, arcname=resolved.name)
    return ToolResult(content=str(destination))


@tools.register(
    capability=ToolCapability(reads_files=True, writes_files=True, concurrency_safe=False),
    permission_policy=_always("archive"),
)
def tar_extract(path: str, target_dir: str = ".", *, context: ToolContext) -> ToolResult:
    """Extract a tar archive."""
    archive = _resolve_path(context, path)
    destination = _resolve_path(context, target_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(destination)
    return ToolResult(content=str(destination))


@tools.register(
    capability=ToolCapability(reads_files=True, writes_files=True, concurrency_safe=False),
    permission_policy=_always("archive"),
)
def zip_create(target: str, paths: list, *, context: ToolContext) -> ToolResult:
    """Create a zip archive."""
    destination = _resolve_path(context, target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as archive:
        for item in paths:
            resolved = _resolve_path(context, item)
            archive.write(resolved, arcname=resolved.name)
    return ToolResult(content=str(destination))


@tools.register(
    capability=ToolCapability(reads_files=True, writes_files=True, concurrency_safe=False),
    permission_policy=_always("archive"),
)
def zip_extract(path: str, target_dir: str = ".", *, context: ToolContext) -> ToolResult:
    """Extract a zip archive."""
    archive = _resolve_path(context, path)
    destination = _resolve_path(context, target_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as handle:
        handle.extractall(destination)
    return ToolResult(content=str(destination))


# --- CSV / text helpers ----------------------------------------------------


@tools.register()
def csv_parse(text: str) -> ToolResult:
    """Parse CSV text into rows."""
    reader = csv.DictReader(io.StringIO(text))
    return ToolResult(content=_json_dump(list(reader)))


@tools.register()
def csv_create(headers: list, rows: list) -> ToolResult:
    """Create CSV text from rows."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(headers))
    writer.writeheader()
    writer.writerows(rows)
    return ToolResult(content=output.getvalue())


@tools.register()
def uuid_generate() -> ToolResult:
    """Generate a UUID."""
    return ToolResult(content=str(uuid.uuid4()))


@tools.register()
def text_sort(text: str) -> ToolResult:
    """Sort lines of text."""
    return ToolResult(content="\n".join(sorted(text.splitlines())))


@tools.register()
def text_dedupe(text: str) -> ToolResult:
    """Remove duplicate lines from text."""
    seen: list[str] = []
    for line in text.splitlines():
        if line not in seen:
            seen.append(line)
    return ToolResult(content="\n".join(seen))


@tools.register()
def text_join(items: list, separator: str = "\n") -> ToolResult:
    """Join items with a separator."""
    return ToolResult(content=separator.join(str(item) for item in items))


@tools.register()
def line_count(text: str) -> ToolResult:
    """Count lines in text."""
    return ToolResult(content=str(len(text.splitlines())))


@tools.register()
def random_string(length: int = 16) -> ToolResult:
    """Generate a random alphanumeric string."""
    alphabet = string.ascii_letters + string.digits
    return ToolResult(content="".join(random.choice(alphabet) for _ in range(length)))


# --- tasks / subagents / VCS / code ---------------------------------------


@tools.register(
    capability=ToolCapability(writes_files=True, concurrency_safe=False),
    permission_policy=_always("todo_write"),
)
def todo_write(items: list, *, context: ToolContext) -> ToolResult:
    """Persist todo items to the task tracker."""
    workspace = context.services.settings.workspace
    saved = []
    for item in items:
        normalized = normalize_task_tracker_item(item)
        if normalized is None:
            continue
        context.services.task_tracker.add_task(
            title=normalized["title"],
            note=normalized["note"],
            workspace=workspace,
        )
        saved.append(normalized["title"])
    return ToolResult(content=_json_dump({"saved": saved}))


@tools.register(
    capability=ToolCapability(concurrency_safe=False),
    permission_policy=PermissionPolicy(kind="task"),
)
def task(
    prompt: str,
    mode: str = "general",
    execution_mode: str = "foreground",
    write_scope: list | None = None,
    depends_on: list | None = None,
    *,
    context: ToolContext,
) -> ToolResult:
    """Submit a single subagent task to the task graph scheduler."""
    submitted = context.services.collaboration.submit_task(
        services=context.services,
        prompt=prompt,
        parent_thread_id=context.thread_id,
        current_mode=context.mode,
        mode_hint=mode,
        execution_mode=execution_mode,
        write_scope=_normalize_write_scope(write_scope),
        depends_on=[str(item) for item in list(depends_on or []) if str(item).strip()],
    )
    payload = {"submitted": submitted}
    return ToolResult(content=_json_dump(payload), structured=payload, metadata=payload)


@tools.register(
    capability=ToolCapability(concurrency_safe=False),
    permission_policy=PermissionPolicy(kind="plan_tasks"),
)
def plan_tasks(items: list, edges: list | None = None, *, context: ToolContext) -> ToolResult:
    """Submit multiple planned tasks and dependencies to the task graph."""
    normalized_items: list[dict[str, Any]] = []
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        payload["write_scope"] = _normalize_write_scope(
            payload.get("write_scope") or payload.get("paths")
        )
        normalized_items.append(payload)
    planned = context.services.collaboration.submit_plan(
        services=context.services,
        items=normalized_items,
        edges=[dict(item) for item in list(edges or []) if isinstance(item, dict)],
        parent_thread_id=context.thread_id,
        current_mode=context.mode,
    )
    return ToolResult(content=_json_dump(planned), structured=planned, metadata=planned)


@tools.register(
    capability=ToolCapability(concurrency_safe=False, long_running=True),
    permission_policy=PermissionPolicy(kind="run_ready_tasks"),
)
def run_ready_tasks(limit: int = 1, allow_parallel: bool = False, *, context: ToolContext) -> ToolResult:
    """Execute ready scheduled tasks in foreground serial order."""
    collaboration = context.services.collaboration
    ready_workers = collaboration.ready_workers(context.services)
    if not ready_workers:
        payload = {"executed": [], "ready_count": 0, "parallel_eligible": []}
        return ToolResult(content=_json_dump(payload), structured=payload, metadata=payload)
    executed: list[dict[str, Any]] = []
    parallel_eligible: list[str] = []
    selected = ready_workers[: max(1, int(limit or 1))]
    running_snapshot: list[Any] = []
    for worker in ready_workers:
        if collaboration.can_run_in_parallel(worker, running_snapshot):
            parallel_eligible.append(worker.worker_id)
            running_snapshot.append(worker)
    for worker in selected:
        if worker.execution_mode == "background":
            context.services.collaboration.queue_worker(
                services=context.services,
                worker_id=worker.worker_id,
            )
            executed.append(
                {
                    "worker_id": worker.worker_id,
                    "node_id": worker.node_id,
                    "status": worker.status,
                    "execution_mode": worker.execution_mode,
                    "skipped": "background_execution_not_implemented",
                }
            )
            continue
        result = _run_worker_until_complete(worker.worker_id, context=context)
        worker_state = collaboration.get_worker(worker.worker_id)
        executed.append(
            {
                "worker_id": worker.worker_id,
                "node_id": worker.node_id,
                "agent_name": worker.agent_name,
                "task_type": worker.task_type,
                "status": worker_state.status,
                "summary": worker_state.summary or getattr(result, "final_text", None),
                "error": worker_state.error or getattr(result, "error", None),
            }
        )
    payload = {
        "executed": executed,
        "ready_count": len(ready_workers),
        "parallel_eligible": parallel_eligible if allow_parallel else [],
    }
    return ToolResult(content=_json_dump(payload), structured=payload, metadata=payload)


@tools.register(
    capability=ToolCapability(shell=True, concurrency_safe=False),
    permission_policy=_always("git"),
)
def git(command: str, timeout: int = 60, *, context: ToolContext) -> ToolResult:
    """Run a git command."""
    code, stdout, stderr = run_command_sync("git " + command, context.cwd, timeout)
    return ToolResult(ok=code == 0, content=_json_dump({"return_code": code, "stdout": stdout, "stderr": stderr}))


@tools.register(capability=ToolCapability(reads_files=True))
def find_symbols(path: str, *, context: ToolContext) -> ToolResult:
    """Find Python symbols in a file."""
    resolved = _resolve_path(context, path)
    tree = ast.parse(_read_text(resolved))
    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            results.append({"name": node.name, "line": node.lineno, "type": type(node).__name__})
    return ToolResult(content=_json_dump(results))


@tools.register(capability=ToolCapability(reads_files=True))
def find_references(
    symbol: str,
    path: str = ".",
    max_matches: int = 100,
    *,
    context: ToolContext,
) -> ToolResult:
    """Search for a symbol across files."""
    return grep_files(
        pattern=re.escape(symbol),
        path=path,
        max_matches=max_matches,
        context=context,
    )


@tools.register(capability=ToolCapability(reads_files=True))
def get_ast_info(path: str, *, context: ToolContext) -> ToolResult:
    """Return Python AST info for a file."""
    resolved = _resolve_path(context, path)
    tree = ast.parse(_read_text(resolved))
    return ToolResult(content=ast.dump(tree, indent=2))


@tools.register(capability=ToolCapability(reads_files=True))
def code_review(path: str, *, context: ToolContext) -> ToolResult:
    """Run simple heuristic code review checks."""
    resolved = _resolve_path(context, path)
    text = _read_text(resolved)
    findings: list[str] = []
    if "TODO" in text:
        findings.append("Found TODO markers.")
    if "print(" in text and resolved.suffix == ".py":
        findings.append("Found print() statements.")
    if "except Exception:" in text:
        findings.append("Found broad exception handler.")
    if not findings:
        findings.append("No obvious heuristic findings.")
    return ToolResult(content="\n".join(findings))


@tools.register(capability=ToolCapability(reads_files=True))
def diff_viewer(
    path_a: str | None = None,
    path_b: str | None = None,
    left: str = "",
    right: str = "",
    *,
    context: ToolContext,
) -> ToolResult:
    """Show a unified diff between two texts or files."""
    if path_a and path_b:
        left_lines = _read_text(_resolve_path(context, path_a)).splitlines()
        right_lines = _read_text(_resolve_path(context, path_b)).splitlines()
    else:
        left_lines = left.splitlines()
        right_lines = right.splitlines()
    diff = difflib.unified_diff(left_lines, right_lines, lineterm="")
    return ToolResult(content="\n".join(diff))


@tools.register(
    capability=ToolCapability(shell=True, concurrency_safe=False),
    permission_policy=_always("test_runner"),
)
def test_runner(
    command: str = "pytest -q",
    timeout: int = 300,
    *,
    context: ToolContext,
) -> ToolResult:
    """Run the test suite or a custom test command."""
    return run_command(command=command, timeout=timeout, context=context)


@tools.register()
def load_skill(name: str, *, context: ToolContext) -> ToolResult:
    """Load the text of an installed skill."""
    body = context.services.skills.load_skill(name)
    return ToolResult(content=body)


# ---------------------------------------------------------------------------
# MCP tools — schema comes from the server, not Python types, so these are
# constructed dynamically and can't use the decorator.
# ---------------------------------------------------------------------------


def _make_mcp_tool(server_name: str, tool_name: str, description: str, schema: dict) -> ToolSpec:
    def _executor(arguments: dict, context: ToolContext) -> ToolResult:
        result = context.services.mcp.call_tool(server_name, tool_name, arguments)
        content = result.get("content", [])
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return ToolResult(content="\n".join(text_parts) or _json_dump(result))

    resolved_name = f"mcp.{server_name}.{tool_name}"
    capability = ToolCapability(network=True, concurrency_safe=False)
    policy = _always("mcp")
    return ToolSpec(
        name=resolved_name,
        description=enrich_tool_description(
            name=resolved_name,
            base_description=description or f"MCP tool {tool_name} from {server_name}.",
            capability=capability,
            permission_policy=policy,
        ),
        input_schema=enrich_input_schema(
            resolved_name,
            schema or {"type": "object", "properties": {}},
        ),
        capability=capability,
        permission_policy=policy,
        validator=_passthrough_validator,
        executor=_executor,
    )


def build_builtin_registry(services: object) -> ToolRegistry:
    registry = ToolRegistry(tools.specs())
    try:
        for server in services.mcp.list_servers():
            for tool in services.mcp.list_tools(server["name"]):
                registry.add(
                    _make_mcp_tool(
                        server_name=server["name"],
                        tool_name=str(tool.get("name", "")),
                        description=str(tool.get("description", "")),
                        schema=dict(tool.get("inputSchema", {}) or {"type": "object"}),
                    )
                )
    except Exception:
        pass
    return registry
