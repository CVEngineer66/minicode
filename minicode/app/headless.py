from __future__ import annotations

import argparse
import json
import sys

from ..features.sessions import format_session_time
from ..runtime.runner import run_turn
from .bootstrap import bootstrap_services


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniCode Next headless runner")
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--resume", dest="thread_id")
    parser.add_argument("--list-sessions", action="store_true")
    parser.add_argument("--decision-json")
    parser.add_argument(
        "--mode",
        choices=["default", "auto", "bypass", "plan"],
        default=None,
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Stream events and the final result as JSONL (one object per line).",
    )
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--cwd", default=".")
    return parser


def _emit_jsonl(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False, default=str), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    services = bootstrap_services(args.cwd)

    if args.list_sessions:
        sessions = services.sessions.list_sessions(
            workspace=services.settings.workspace
        )
        if args.jsonl:
            for s in sessions:
                _emit_jsonl(
                    {
                        "type": "session",
                        "thread_id": s.thread_id,
                        "updated_at": s.updated_at,
                        "title": s.title,
                    }
                )
            return 0
        if not sessions:
            print("No saved sessions found.")
            return 0
        for session in sessions:
            print(f"{session.thread_id}\t{format_session_time(session.updated_at)}\t{session.title}")
        return 0

    resume = json.loads(args.decision_json) if args.decision_json else None
    thread_id = args.thread_id
    if thread_id:
        # Scope the id/prefix/"latest" resolution to the current workspace so
        # an unknown or cross-workspace id can't silently create a phantom
        # session via ensure_thread.
        resolved, error = services.sessions.resolve_thread_id(
            thread_id, workspace=services.settings.workspace
        )
        if error is not None:
            print(error, file=sys.stderr)
            return 2
        thread_id = resolved

    event_sink = None
    if args.jsonl:
        def event_sink(event):  # type: ignore[no-redef]
            _emit_jsonl(
                {
                    "type": "event",
                    "kind": event.kind,
                    "payload": event.payload,
                    "timestamp": event.timestamp,
                }
            )

    result = run_turn(
        services=services,
        prompt=args.prompt,
        thread_id=thread_id,
        resume=resume,
        mode=args.mode,
        max_steps=args.max_steps,
        event_sink=event_sink,
    )

    if args.jsonl:
        _emit_jsonl(
            {
                "type": "result",
                "thread_id": result.thread_id,
                "final_text": result.final_text,
                "interrupt": result.interrupt,
                "await_user": result.await_user,
                "error": result.error,
            }
        )
        return 0 if result.error is None else 1

    if result.interrupt:
        print(
            json.dumps(
                {"thread_id": result.thread_id, "__interrupt__": result.interrupt},
                ensure_ascii=False,
            )
        )
        return 0
    if result.final_text:
        print(result.final_text)
        return 0
    if result.error:
        print(result.error, file=sys.stderr)
        return 1
    return 0
