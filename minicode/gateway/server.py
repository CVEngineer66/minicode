from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import parse

from minicode.app.bootstrap import bootstrap_services
from minicode.runtime.runner import run_turn

_log = logging.getLogger("minicode.gateway")


def _json_response(
    handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]
) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    handler.wfile.write(body)


def _serialize_event(event: Any) -> dict[str, Any]:
    return {
        "kind": getattr(event, "kind", ""),
        "payload": getattr(event, "payload", {}) or {},
        "timestamp": getattr(event, "timestamp", 0.0),
    }


def _serialize_result(result: Any) -> dict[str, Any]:
    return {
        "thread_id": result.thread_id,
        "final_text": result.final_text,
        "interrupt": result.interrupt,
        "await_user": result.await_user,
        "error": result.error,
        "events": [_serialize_event(event) for event in result.events],
    }


def _serialize_health(services: Any) -> dict[str, Any]:
    settings = services.settings
    payload = {
        "status": "ok",
        "workspace": settings.workspace,
        "provider": settings.provider,
        "model": settings.model,
    }
    auto = getattr(services, "auto", None)
    if auto is not None and hasattr(auto, "get_mode"):
        payload["mode"] = getattr(auto.get_mode(), "value", settings.auto_mode)
    else:
        payload["mode"] = settings.auto_mode
    return payload


def _serialize_sessions(services: Any) -> dict[str, Any]:
    sessions = services.sessions.list_sessions(workspace=services.settings.workspace)
    return {
        "sessions": [
            {
                "thread_id": session.thread_id,
                "updated_at": session.updated_at,
                "title": session.title,
                "model": session.model,
            }
            for session in sessions
        ]
    }


class _GatewayHandler(BaseHTTPRequestHandler):
    server_version = "MiniCode/0.2"
    protocol_version = "HTTP/1.1"
    MAX_BODY_BYTES = 1 * 1024 * 1024

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        _log.debug("http %s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = parse.urlparse(self.path)
            path = parsed.path
            if path == "/health":
                _json_response(self, 200, _serialize_health(self.server.services))
                return
            if path == "/sessions":
                _json_response(self, 200, _serialize_sessions(self.server.services))
                return
            _json_response(self, 404, {"error": f"Not found: {path}"})
        except BaseException as exc:  # noqa: BLE001
            _log.exception("gateway GET failed")
            _json_response(self, 500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        parsed = parse.urlparse(self.path)
        path = parsed.path
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            _json_response(self, 400, {"error": str(exc)})
            return
        except BaseException as exc:  # noqa: BLE001
            _log.exception("gateway POST body parse failed")
            _json_response(self, 500, {"error": str(exc)})
            return

        if path == "/turn":
            self._handle_turn(payload)
            return
        if path == "/resume":
            self._handle_resume(payload)
            return
        _json_response(self, 404, {"error": f"Not found: {path}"})

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > self.MAX_BODY_BYTES:
            raise ValueError("request body too large")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("expected JSON object")
        return payload

    def _handle_turn(self, payload: dict[str, Any]) -> None:
        kwargs = {
            "services": self.server.services,
            "prompt": payload.get("prompt"),
            "thread_id": payload.get("thread_id"),
            "mode": payload.get("mode"),
            "max_steps": int(payload.get("max_steps", 40)),
        }
        try:
            result = run_turn(**kwargs)
        except BaseException as exc:  # noqa: BLE001
            _log.exception("gateway /turn failed")
            _json_response(self, 500, {"error": str(exc)})
            return
        _json_response(self, 200, _serialize_result(result))

    def _handle_resume(self, payload: dict[str, Any]) -> None:
        thread_id = payload.get("thread_id")
        decision = payload.get("decision")
        if not thread_id or decision is None:
            _json_response(self, 400, {"error": "thread_id and decision are required"})
            return
        kwargs = {
            "services": self.server.services,
            "thread_id": thread_id,
            "resume": decision if isinstance(decision, dict) else {"decision": decision},
        }
        try:
            result = run_turn(**kwargs)
        except BaseException as exc:  # noqa: BLE001
            _log.exception("gateway /resume failed")
            _json_response(self, 500, {"error": str(exc)})
            return
        _json_response(self, 200, _serialize_result(result))


class GatewayServer:
    """Threaded HTTP gateway wrapping a bootstrapped AppServices instance."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7681,
        cwd: str = ".",
        services: Any | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.services = services or bootstrap_services(cwd)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, *, block: bool = True) -> None:
        self._httpd = ThreadingHTTPServer((self.host, self.port), _GatewayHandler)
        self._httpd.services = self.services  # type: ignore[attr-defined]
        self.port = int(self._httpd.server_address[1])
        _log.info("gateway listening on %s", self.url)
        if block:
            try:
                self._httpd.serve_forever()
            except KeyboardInterrupt:
                _log.info("gateway interrupted")
            finally:
                self.stop()
            return
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
