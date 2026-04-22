from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    step_count: int
    max_steps: int
    saw_tool_result: bool
    empty_response_retries: int
    tool_error_count: int
    final_text: str | None
    error: str | None
    await_user: bool
    thread_id: str
    prompt_context: str
    prompt_context_static: str
    prompt_context_dynamic: str
    user_query: str
    mode: str
