from __future__ import annotations

import threading
import time
from typing import Any, Callable

from .client import McpClient, McpConnectionError, McpTimeoutError


class McpClientPool:
    """Cache + health check for long-lived MCP client connections.

    Boundaries:
    - Each server name maps to at most one live McpClient.
    - `max_idle_s` closes clients unused for too long (default 300s).
    - `health_check` verifies liveness and reconnects on failure.
    - Caller-provided factory encapsulates server config lookup.
    """

    def __init__(
        self,
        factory: Callable[[str], McpClient],
        *,
        max_idle_s: float = 300.0,
    ) -> None:
        self._factory = factory
        self._clients: dict[str, McpClient] = {}
        self._last_used: dict[str, float] = {}
        self._lock = threading.Lock()
        self.max_idle_s = max_idle_s

    def get(self, name: str) -> McpClient:
        with self._lock:
            client = self._clients.get(name)
            if client is not None and client.is_alive():
                self._last_used[name] = time.time()
                return client
            if client is not None:
                self._drop(name)
            try:
                client = self._factory(name)
            except McpConnectionError:
                raise
            self._clients[name] = client
            self._last_used[name] = time.time()
            return client

    def health_check(self, name: str) -> bool:
        try:
            client = self.get(name)
            client.list_tools()
            return True
        except (McpConnectionError, McpTimeoutError):
            self.drop(name)
            return False

    def drop(self, name: str) -> None:
        with self._lock:
            self._drop(name)

    def close_all(self) -> None:
        with self._lock:
            for name in list(self._clients):
                self._drop(name)

    def reap_idle(self) -> list[str]:
        now = time.time()
        evicted: list[str] = []
        with self._lock:
            for name, ts in list(self._last_used.items()):
                if now - ts > self.max_idle_s:
                    self._drop(name)
                    evicted.append(name)
        return evicted

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": list(self._clients.keys()),
                "last_used": dict(self._last_used),
            }

    # --- internal ---
    def _drop(self, name: str) -> None:
        client = self._clients.pop(name, None)
        self._last_used.pop(name, None)
        if client is not None:
            try:
                client.close()
            except BaseException:
                pass
