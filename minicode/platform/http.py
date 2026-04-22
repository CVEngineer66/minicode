from __future__ import annotations

import json
from typing import Any
from urllib import parse, request


def http_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: Any = None,
    timeout: int = 30,
) -> tuple[int, str]:
    data = None
    final_headers = dict(headers or {})
    if body is not None:
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode("utf-8")
            final_headers.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            data = body.encode("utf-8")
        else:
            data = body
    req = request.Request(url=url, data=data, headers=final_headers, method=method.upper())
    with request.urlopen(req, timeout=timeout) as response:
        return int(response.status), response.read().decode("utf-8", errors="replace")


def simple_web_search(query: str, timeout: int = 30) -> str:
    encoded = parse.quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={encoded}"
    status, content = http_request("GET", url, timeout=timeout)
    if status >= 400:
        raise RuntimeError(f"search failed with status {status}")
    lines = [line.strip() for line in content.splitlines() if "result__a" in line][:5]
    return "\n".join(lines) if lines else content[:2000]
