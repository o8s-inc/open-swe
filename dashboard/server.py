"""Dashboard agent — LangGraph server for ops-dashboard AI chat."""

from __future__ import annotations

import os
from typing import Annotated, Any, Sequence

from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .tools import query_argo, query_litellm, query_mimir, query_tracker

# ── Model ──────────────────────────────────────────────────────────────────
# Uses LiteLLM proxy if OPENAI_API_BASE is set (standard Open SWE pattern)
_model_name = os.environ.get("DASHBOARD_MODEL", "gpt-4o-mini")

llm = ChatOpenAI(
    model=_model_name,
    # OPENAI_API_BASE is picked up automatically by langchain-openai
    temperature=0,
    streaming=False,
)

TOOLS = [query_tracker, query_litellm, query_mimir, query_argo]
llm_with_tools = llm.bind_tools(TOOLS)

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an operations assistant for the o8s.ai platform.
You have access to tools that let you query the ticket tracker, LLM cost data,
cluster metrics, and Argo build pipelines.

Be concise and factual. Use tools to answer questions about:
- Open/in-progress tickets and their assignees
- LLM costs (today, week, month) broken down by model or agent
- Kubernetes cluster health (nodes, CPU, memory, top pods)
- Build pipeline status (Argo Workflows — succeeded, failed, running)

When you don't know something, use a tool to find out.
Format numbers clearly: use GB/MB for memory, seconds/minutes for durations,
$X.XXXX for costs.
"""

# ── State ──────────────────────────────────────────────────────────────────


class State(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]


# ── Graph nodes ────────────────────────────────────────────────────────────


def call_model(state: State) -> dict[str, Any]:
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)] + list(
        state["messages"]
    )
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Build graph ────────────────────────────────────────────────────────────

tool_node = ToolNode(TOOLS)

graph_builder = StateGraph(State)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()


def get_graph() -> Any:
    """Entry point for LangGraph server registration."""
    return graph
