from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

from .types import GraphEvent


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[GraphEvent], None]]] = defaultdict(list)

    def subscribe(self, kind: str, handler: Callable[[GraphEvent], None]) -> None:
        self._handlers[kind].append(handler)

    def emit(self, kind: str, **payload: object) -> GraphEvent:
        event = GraphEvent(kind=kind, payload=dict(payload), timestamp=time.time())
        for handler in self._handlers.get(kind, []):
            handler(event)
        for handler in self._handlers.get("*", []):
            handler(event)
        return event
