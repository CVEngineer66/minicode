from __future__ import annotations

from typing import Callable

from .types import HookEvent, HookHandler, HookRegistration


class HookRegistry:
    """In-memory hook handler registry keyed by HookEvent."""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[HookRegistration]] = {e: [] for e in HookEvent}

    def register(
        self,
        event: HookEvent,
        handler: HookHandler,
        description: str = "",
    ) -> Callable[[], None]:
        reg = HookRegistration(event=event, handler=handler, description=description)
        self._hooks[event].append(reg)

        def unregister() -> None:
            if reg in self._hooks[event]:
                self._hooks[event].remove(reg)

        return unregister

    def registrations(self, event: HookEvent) -> list[HookRegistration]:
        return list(self._hooks[event])

    def all(self) -> list[HookRegistration]:
        return [r for bucket in self._hooks.values() for r in bucket]

    def clear(self) -> None:
        for bucket in self._hooks.values():
            bucket.clear()
