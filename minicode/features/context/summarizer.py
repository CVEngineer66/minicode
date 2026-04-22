from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .token_estimator import estimate_tokens

_EDIT_TOOLS = frozenset({"edit_file", "write_file", "modify_file", "patch_file", "multi_edit"})
_READ_TOOLS = frozenset({"read_file", "list_files", "grep_files", "file_tree"})
_SEARCH_TOOLS = frozenset({"grep_files", "find_symbols", "find_references", "web_search", "web_fetch"})
_COMMAND_TOOLS = frozenset({"run_command", "execute_command", "bash"})

_CODE_FENCE_RE = re.compile(r"```[\w]*\n(.{20,300}?)```", re.DOTALL)
_DECISION_KEYWORDS = re.compile(
    r"(?:decided|decision|chose|chosen|will use|using|switching to|"
    r"implemented|fixed|resolved|refactored|migrated|upgraded|"
    r"recommend|should|must|need to|going to|plan to|"
    r"approach:|strategy:|solution:|conclusion:)",
    re.IGNORECASE,
)


@dataclass
class ExtractedInfo:
    user_intents: list[str] = field(default_factory=list)
    file_paths: set[str] = field(default_factory=set)
    key_tool_results: list[str] = field(default_factory=list)
    assistant_conclusions: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    code_snippets: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)


def _extract(messages: list[dict[str, Any]]) -> ExtractedInfo:
    info = ExtractedInfo()
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "user" and content.strip():
            preview = content.strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            info.user_intents.append(preview)
        elif role == "assistant" and content.strip():
            text = content.strip()
            for sentence in text.replace("\n", " ").split(". "):
                if _DECISION_KEYWORDS.search(sentence):
                    decision = sentence.strip()[:180]
                    if decision and decision not in info.decisions:
                        info.decisions.append(decision)
            for match in _CODE_FENCE_RE.finditer(text):
                snippet = match.group(1).strip()
                if len(snippet) >= 20 and len(info.code_snippets) < 5:
                    info.code_snippets.append(snippet[:300])
            info.assistant_conclusions.append(text[:200].replace("\n", " "))
        elif role == "assistant_tool_call":
            tool_name = msg.get("toolName", "unknown")
            info.tool_names.append(tool_name)
            inp = msg.get("input", {}) or {}
            if tool_name in _EDIT_TOOLS:
                path = inp.get("path") or inp.get("filePath", "")
                if path:
                    info.file_paths.add(path)
            if tool_name in _SEARCH_TOOLS:
                pattern = inp.get("pattern") or inp.get("query", "")
                if pattern:
                    info.file_paths.add(f"search:{pattern[:80]}")
            if tool_name in _COMMAND_TOOLS:
                cmd = inp.get("command", "")
                if cmd:
                    parts = cmd.split()
                    if parts:
                        info.key_tool_results.append(f"ran: {parts[0]}")
        elif role == "tool_result":
            tool_name = msg.get("toolName", "")
            if msg.get("isError"):
                preview = content.strip()[:150].replace("\n", " ")
                info.key_tool_results.append(f"ERROR({tool_name}): {preview}")
            elif tool_name in _EDIT_TOOLS and content.strip():
                preview = content.strip()[:100].replace("\n", " ")
                info.key_tool_results.append(f"{tool_name} ok: {preview}")
            elif tool_name in _READ_TOOLS and content.strip():
                first_line = content.strip().split("\n")[0][:100]
                if "/" in first_line or "\\" in first_line:
                    info.file_paths.add(first_line.strip())
    return info


def _layered_summary(info: ExtractedInfo, max_tokens: int = 2000) -> str:
    lines: list[str] = []
    budgets = [0.35, 0.20, 0.15, 0.15, 0.10, 0.05]

    def used() -> int:
        return estimate_tokens("\n".join(lines))

    if info.user_intents:
        budget = int(max_tokens * budgets[0])
        lines.append("## User requests:")
        for intent in info.user_intents[:12]:
            if used() > budget:
                break
            lines.append(f"- {intent}")
    if info.decisions or info.file_paths:
        budget = int(max_tokens * sum(budgets[:2]))
        if info.decisions:
            lines.append("## Key decisions:")
            for dec in info.decisions[:8]:
                if used() > budget:
                    break
                lines.append(f"- {dec}")
        if info.file_paths:
            real = sorted(p for p in info.file_paths if not p.startswith("search:"))
            searches = sorted(p[8:] for p in info.file_paths if p.startswith("search:"))
            path_line = f"## Files: {', '.join(real[:20])}"
            if len(real) > 20:
                path_line += f" (+{len(real)-20} more)"
            if searches:
                path_line += f"\n## Searched: {', '.join(searches[:5])}"
            if estimate_tokens("\n".join(lines) + path_line) <= budget:
                lines.append(path_line)
    if info.key_tool_results:
        budget = int(max_tokens * sum(budgets[:3]))
        lines.append("## Key results:")
        for result in info.key_tool_results[:15]:
            if used() > budget:
                break
            lines.append(f"- {result}")
    if info.assistant_conclusions:
        budget = int(max_tokens * sum(budgets[:4]))
        lines.append("## Conclusions:")
        for conc in info.assistant_conclusions[:8]:
            if used() > budget:
                break
            lines.append(f"- {conc}")
    if info.code_snippets:
        budget = int(max_tokens * sum(budgets[:5]))
        lines.append("## Code patterns:")
        for snippet in info.code_snippets[:3]:
            block = f"```\n{snippet}\n```"
            if estimate_tokens("\n".join(lines) + block) > budget:
                break
            lines.append(block)
    if info.tool_names:
        counts = Counter(info.tool_names)
        summary = ", ".join(f"{n}×{c}" if c > 1 else n for n, c in counts.most_common())
        lines.append(f"## Tools: {summary}")
    return "\n".join(lines)


def summarize_removed(messages: list[dict[str, Any]], max_tokens: int = 2000) -> str:
    if not messages:
        return ""
    return _layered_summary(_extract(messages), max_tokens)


def compress_tool_pair(call: dict[str, Any], result: dict[str, Any]) -> str:
    tool_name = call.get("toolName", "unknown")
    inp = call.get("input", {}) or {}
    result_content = result.get("content", "") or ""
    if result.get("isError"):
        return f"[Tool {tool_name} ERROR: {result_content.strip()[:200].replace(chr(10), ' ')}]"
    if tool_name in _EDIT_TOOLS:
        path = inp.get("path") or inp.get("filePath", "unknown")
        if tool_name == "multi_edit":
            edits = inp.get("edits", []) or []
            return f"[Edited {path}: {len(edits)} changes applied]"
        return f"[Edited {path}: ok]"
    if tool_name in _READ_TOOLS:
        path = inp.get("path") or inp.get("filePath", "")
        if path:
            line_count = result_content.count("\n") + 1
            return f"[Read {path}: {line_count} lines]"
        return f"[{tool_name}: completed]"
    if tool_name in _SEARCH_TOOLS:
        pattern = inp.get("pattern") or inp.get("query", "")
        matches = [l for l in result_content.split("\n") if l.strip() and not l.startswith("#")]
        return f"[Searched '{pattern[:50]}': {len(matches)} results]"
    if tool_name in _COMMAND_TOOLS:
        cmd = inp.get("command", "")
        name = cmd.split()[0] if cmd.split() else "command"
        exit_info = ""
        for line in result_content.split("\n"):
            if "exit code" in line.lower():
                exit_info = f" ({line.strip()[:50]})"
                break
        return f"[Ran {name}{exit_info}]"
    brief = result_content.strip()[:100].replace("\n", " ")
    return f"[{tool_name}: {brief}]" if brief else f"[{tool_name}: completed]"
