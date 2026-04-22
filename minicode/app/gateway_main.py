from __future__ import annotations

import argparse
import logging

from minicode.gateway import GatewayServer
from minicode.platform.logging import setup_logging
from minicode.platform.paths import resolve_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniCode Next HTTP gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7681)
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--structured-logs", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolve_paths(args.cwd)
    setup_logging(
        paths.logs_dir,
        level=args.log_level,
        structured=args.structured_logs,
        log_name="gateway.log",
    )
    log = logging.getLogger("minicode.gateway")
    log.info("starting gateway on %s:%d", args.host, args.port)
    server = GatewayServer(host=args.host, port=args.port, cwd=args.cwd)
    server.start(block=True)
    return 0
