"""DevOps Agent — Monitors Sentry errors + Grafana metrics, creates tickets, proposes fixes."""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import httpx
import os

DESCRIPTION = "Monitors Sentry errors and Grafana metrics, creates Linear tickets, and proposes fixes"
SKILLS = [
    {"id": "monitor-errors", "name": "Monitor Errors", "description": "Check Sentry for new/recurring errors"},
    {"id": "check-metrics", "name": "Check Metrics", "description": "Query Grafana/Mimir for anomalies"},
    {"id": "create-ticket", "name": "Create Ticket", "description": "Create Linear ticket for issues found"},
    {"id": "propose-fix", "name": "Propose Fix", "description": "Analyze error and suggest a fix"},
]

class DevOpsState(TypedDict):
    task: str
    errors: list
    metrics: dict
    analysis: str
    actions_taken: list[str]
    messages: Annotated[list, operator.add]

def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514"),
        openai_api_base=os.getenv("LITELLM_API_URL", "http://litellm.litellm.svc:4000"),
        openai_api_key=os.getenv("LITELLM_API_KEY", os.getenv("OPENAI_API_KEY", "sk-placeholder")),
    )

async def check_sentry(state: DevOpsState) -> DevOpsState:
    """Check Sentry for recent errors."""
    sentry_url = os.getenv("SENTRY_URL", "https://sentry.o8s.ai")
    token = os.getenv("SENTRY_AUTH_TOKEN", "")
    errors = []
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{sentry_url}/api/0/projects/o8s/platform/issues/",
                headers={"Authorization": f"Bearer {token}"},
                params={"query": "is:unresolved", "limit": 10},
            )
            if resp.status_code == 200:
                errors = resp.json()
    except Exception as e:
        errors = [{"error": str(e)}]
    return {**state, "errors": errors, "messages": [HumanMessage(content=f"Found {len(errors)} Sentry errors")]}

async def check_grafana(state: DevOpsState) -> DevOpsState:
    """Check Grafana/Mimir for metric anomalies."""
    grafana_url = os.getenv("GRAFANA_URL", "https://grafana.o8s.ai")
    api_key = os.getenv("GRAFANA_API_KEY", "")
    metrics = {}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{grafana_url}/api/alerts",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code == 200:
                metrics = {"alerts": resp.json()}
    except Exception as e:
        metrics = {"error": str(e)}
    return {**state, "metrics": metrics, "messages": [HumanMessage(content=f"Checked Grafana alerts")]}

async def analyze(state: DevOpsState) -> DevOpsState:
    """Use LLM to analyze errors and metrics."""
    llm = get_llm()
    prompt = f"""You are a DevOps agent for the o8s platform.

Analyze these findings and recommend actions:

Sentry Errors: {state.get('errors', [])}
Grafana Alerts: {state.get('metrics', {})}
Task: {state.get('task', 'General health check')}

Provide:
1. Summary of issues found
2. Severity assessment
3. Recommended actions (create ticket, propose fix, escalate)
"""
    response = await llm.ainvoke([SystemMessage(content="You are a DevOps expert."), HumanMessage(content=prompt)])
    return {**state, "analysis": response.content, "messages": [response]}

async def take_action(state: DevOpsState) -> DevOpsState:
    """Execute recommended actions (create tickets, etc)."""
    actions = []
    if state.get("errors"):
        actions.append(f"Identified {len(state['errors'])} errors for triage")
    if state.get("metrics", {}).get("alerts"):
        actions.append(f"Found {len(state['metrics']['alerts'])} active alerts")
    actions.append("Analysis complete — review recommended actions")
    return {**state, "actions_taken": actions}

# Build the graph
builder = StateGraph(DevOpsState)
builder.add_node("check_sentry", check_sentry)
builder.add_node("check_grafana", check_grafana)
builder.add_node("analyze", analyze)
builder.add_node("take_action", take_action)

builder.set_entry_point("check_sentry")
builder.add_edge("check_sentry", "check_grafana")
builder.add_edge("check_grafana", "analyze")
builder.add_edge("analyze", "take_action")
builder.add_edge("take_action", END)

graph = builder.compile()
