from __future__ import annotations

import threading
import time
from typing import Any, Callable

from minicode.core.events import EventBus

from .registry import HookRegistry
from .types import HookContext, HookEvent, HookHandler, HookRegistration


class HookTimeoutError(RuntimeError):
    pass


def _run_with_timeout(func: Callable[[], Any], timeout_s: float) -> Any:
    """Run func in a worker thread; raise HookTimeoutError if it exceeds timeout.

    Note: this cannot interrupt CPU-bound handlers in pure Python, but it bounds
    the wait so the caller is never blocked longer than timeout_s.
    """
    result: list[Any] = [None]
    error: list[BaseException | None] = [None]

    def target() -> None:
        try:
            result[0] = func()
        except BaseException as exc:
            error[0] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise HookTimeoutError(f"hook exceeded {timeout_s}s")
    if error[0] is not None:
        raise error[0]
    return result[0]


class HookService:
    """Lifecycle hooks with per-handler timeout and error isolation.

    Boundaries:
    - Each handler is bounded by `timeout_s` (default 5s).
    - Handler errors and timeouts are captured and surfaced to ON_ERROR hooks,
      but never break the main flow (ON_ERROR handlers themselves are best-effort).
    - All fire() invocations also emit a corresponding event on the EventBus so
      UI/runtime observers can subscribe without touching the hook registry.
    """

    DEFAULT_TIMEOUT_S = 5.0

    def __init__(
        self,
        bus: EventBus,
        *,
        registry: HookRegistry | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.bus = bus
        self.registry = registry or HookRegistry()
        self.timeout_s = timeout_s if timeout_s is not None else self.DEFAULT_TIMEOUT_S
        self._enabled = True

    # --- registration ---
    def register(
        self,
        event: HookEvent | str,
        handler: HookHandler,
        description: str = "",
    ) -> Callable[[], None]:
        event = event if isinstance(event, HookEvent) else HookEvent(event)
        return self.registry.register(event, handler, description)

    def unregister_all(self) -> None:
        self.registry.clear()

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    # --- firing ---
    def fire(self, event: HookEvent | str, **data: Any) -> list[Any]:
        if not self._enabled:
            return []
        event = event if isinstance(event, HookEvent) else HookEvent(event)
        context = HookContext(event=event, data=dict(data))
        self.bus.emit(f"hook.{event.value}", **data)

        results: list[Any] = []
        errors: list[tuple[HookRegistration, BaseException]] = []
        for reg in self.registry.registrations(event):
            if not reg.enabled:
                continue
            start = time.time()
            try:
                value = _run_with_timeout(lambda r=reg: r.handler(context), self.timeout_s)
                reg.call_count += 1
                reg.last_called = time.time()
                reg.total_duration_ms += int((time.time() - start) * 1000)
                results.append(value)
            except HookTimeoutError as exc:
                reg.timeout_count += 1
                reg.error_count += 1
                errors.append((reg, exc))
                results.append({"error": "timeout", "hook": reg.description})
            except BaseException as exc:
                reg.error_count += 1
                errors.append((reg, exc))
                results.append({"error": str(exc), "hook": reg.description})

        # Best-effort ON_ERROR dispatch for handler failures
        if errors and event is not HookEvent.ON_ERROR:
            self._dispatch_errors(event, errors)
        return results

    def _dispatch_errors(
        self,
        source_event: HookEvent,
        errors: list[tuple[HookRegistration, BaseException]],
    ) -> None:
        payload = [
            {
                "hook": reg.description or getattr(reg.handler, "__name__", "anonymous"),
                "source_event": source_event.value,
                "error": str(exc),
            }
            for reg, exc in errors
        ]
        try:
            self.fire(HookEvent.ON_ERROR, failures=payload)
        except BaseException:
            # ON_ERROR handlers cannot themselves break the outer flow.
            pass

    # --- compatibility with pre-existing EventBus passthrough ---
    def emit(self, kind: str, **payload: Any) -> None:
        self.bus.emit(kind, **payload)

    def subscribe(self, kind: str, handler: Callable[..., None]) -> None:
        self.bus.subscribe(kind, handler)

    # --- introspection ---
    def stats(self, event: HookEvent | None = None) -> dict[str, Any]:
        regs = (
            self.registry.registrations(event) if event is not None else self.registry.all()
        )
        return {
            "total_hooks": len(regs),
            "enabled_hooks": sum(1 for r in regs if r.enabled),
            "total_calls": sum(r.call_count for r in regs),
            "error_count": sum(r.error_count for r in regs),
            "timeout_count": sum(r.timeout_count for r in regs),
            "total_duration_ms": sum(r.total_duration_ms for r in regs),
        }
