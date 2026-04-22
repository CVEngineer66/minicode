from __future__ import annotations

import argparse

from .bootstrap import bootstrap_services
from ..features.sessions import format_session_time
from ..ui.tui import run_tui_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minicode", description="MiniCode")
    parser.add_argument(
        "-c",
        "--continue",
        dest="continue_session",
        action="store_true",
        help="resume the latest session from the current workspace",
    )
    parser.add_argument(
        "--resume",
        dest="resume_thread",
        metavar="THREAD_ID",
        help="resume a specific session by thread id (prefix match)",
    )

    subparsers = parser.add_subparsers(dest="command")

    mcp_parser = subparsers.add_parser("mcp")
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_command")
    mcp_sub.add_parser("list")
    mcp_add = mcp_sub.add_parser("add")
    mcp_add.add_argument("name")
    mcp_add.add_argument("server_command")
    mcp_add.add_argument("args", nargs="*")

    skill_parser = subparsers.add_parser("skills")
    skill_sub = skill_parser.add_subparsers(dest="skills_command")
    skill_sub.add_parser("list")
    skill_install = skill_sub.add_parser("install")
    skill_install.add_argument("path")

    sessions_parser = subparsers.add_parser("sessions")
    sessions_sub = sessions_parser.add_subparsers(dest="sessions_command")
    sessions_sub.add_parser("list")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    services = bootstrap_services(".")

    if args.command == "mcp":
        if args.mcp_command == "list":
            for item in services.mcp.list_servers():
                print(f"{item['name']}\t{item['command']}")
            return 0
        if args.mcp_command == "add":
            services.mcp.add_server(
                name=args.name,
                command=args.server_command,
                args=list(args.args),
                env={},
                cwd=None,
            )
            print(f"Added MCP server {args.name}")
            return 0
    if args.command == "skills":
        if args.skills_command == "list":
            for item in services.skills.list_skills():
                print(f"{item['name']}\t{item['source']}")
            return 0
        if args.skills_command == "install":
            services.skills.install_from_path(args.path)
            print(f"Installed skill from {args.path}")
            return 0
    if args.command == "sessions":
        for item in services.sessions.list_sessions(
            workspace=services.settings.workspace
        ):
            print(f"{item.thread_id}\t{format_session_time(item.updated_at)}\t{item.title}")
        return 0

    initial_thread_id: str | None = None
    if args.resume_thread:
        # Pre-resolve against the current workspace. Without this, an unknown
        # prefix used to leak straight into run_turn where ensure_thread would
        # create a phantom session with the prefix as its thread_id.
        resolved, error = services.sessions.resolve_thread_id(
            args.resume_thread, workspace=services.settings.workspace
        )
        if error is not None:
            print(error)
        else:
            initial_thread_id = resolved
    elif args.continue_session:
        latest = services.sessions.get_latest_session(
            workspace=services.settings.workspace
        )
        if latest is None:
            print("No previous session in this workspace to continue.")
        else:
            initial_thread_id = latest.thread_id

    return run_tui_app(services, initial_thread_id=initial_thread_id)


if __name__ == "__main__":
    raise SystemExit(main())
