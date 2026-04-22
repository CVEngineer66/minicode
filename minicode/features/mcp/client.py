from __future__ import annotations

import json
import select
import subprocess
import sys
import time
from itertools import count
from typing import Any

from .validation import MAX_MCP_PAYLOAD_BYTES, validate_args, validate_command


class McpConnectionError(RuntimeError):
    pass


class McpTimeoutError(RuntimeError):
    pass


class McpClient:
    """JSON-RPC stdio client to an MCP server.

    Boundaries:
    - `connect_timeout_s` (default 5s) bounds `initialize`.
    - `call_timeout_s` (default 60s) bounds every subsequent request.
    - Replies larger than `MAX_MCP_PAYLOAD_BYTES` are rejected.
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        connect_timeout_s: float = 5.0,
        call_timeout_s: float = 60.0,
        validate: bool = True,
    ) -> None:
        if validate:
            validate_command(command)
            validate_args(args)
        self.command = command
        self.args = list(args)
        self.connect_timeout_s = connect_timeout_s
        self.call_timeout_s = call_timeout_s
        self._ids = count(1)
        try:
            self.process = subprocess.Popen(
                [command, *args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=env or None,
            )
        except (OSError, FileNotFoundError) as exc:
            raise McpConnectionError(f"Failed to start MCP server: {exc}") from exc
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise

    # --- core I/O ---
    def _send(self, payload: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        timeout = self.call_timeout_s if timeout_s is None else timeout_s
        try:
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise McpConnectionError(f"MCP stdin failure: {exc}") from exc
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise McpTimeoutError(f"MCP call timed out after {timeout}s")
            line = self._readline_with_timeout(remaining)
            if not line:
                raise McpConnectionError("MCP server closed unexpectedly")
            if len(line.encode("utf-8", errors="ignore")) > MAX_MCP_PAYLOAD_BYTES:
                raise McpConnectionError("MCP payload exceeds size cap")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") == payload.get("id"):
                return message

    def _readline_with_timeout(self, timeout_s: float) -> str:
        stdout = self.process.stdout
        assert stdout is not None
        if sys.platform == "win32":
            # select() on pipes is unsupported on Windows — fall back to blocking read.
            return stdout.readline()
        ready, _, _ = select.select([stdout], [], [], timeout_s)
        if not ready:
            raise McpTimeoutError(f"MCP read timed out after {timeout_s}s")
        return stdout.readline()

    def _initialize(self) -> None:
        init_id = next(self._ids)
        self._send(
            {"jsonrpc": "2.0", "id": init_id, "method": "initialize", "params": {}},
            timeout_s=self.connect_timeout_s,
        )
        # fire-and-forget notification
        assert self.process.stdin is not None
        self.process.stdin.write(
            json.dumps(
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
            )
            + "\n"
        )
        self.process.stdin.flush()

    # --- high level ---
    def list_tools(self) -> list[dict[str, Any]]:
        msg = self._send(
            {"jsonrpc": "2.0", "id": next(self._ids), "method": "tools/list", "params": {}}
        )
        return list(msg.get("result", {}).get("tools", []))

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        msg = self._send(
            {
                "jsonrpc": "2.0",
                "id": next(self._ids),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        return msg.get("result", {})

    def list_resources(self) -> list[dict[str, Any]]:
        msg = self._send(
            {
                "jsonrpc": "2.0",
                "id": next(self._ids),
                "method": "resources/list",
                "params": {},
            }
        )
        return list(msg.get("result", {}).get("resources", []))

    def read_resource(self, uri: str) -> dict[str, Any]:
        msg = self._send(
            {
                "jsonrpc": "2.0",
                "id": next(self._ids),
                "method": "resources/read",
                "params": {"uri": uri},
            }
        )
        return msg.get("result", {})

    def list_prompts(self) -> list[dict[str, Any]]:
        msg = self._send(
            {"jsonrpc": "2.0", "id": next(self._ids), "method": "prompts/list", "params": {}}
        )
        return list(msg.get("result", {}).get("prompts", []))

    def get_prompt(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        msg = self._send(
            {
                "jsonrpc": "2.0",
                "id": next(self._ids),
                "method": "prompts/get",
                "params": {"name": name, "arguments": arguments},
            }
        )
        return msg.get("result", {})

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        except OSError:
            pass
