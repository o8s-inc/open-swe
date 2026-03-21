"""Ops Agent — Checks cluster health, triages alerts, runs playbooks."""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import httpx
import os
import subprocess

DESCRIPTION = "Checks Kubernetes cluster health, triages alerts, and suggests operational actions"
SKILLS = [
    {"id": "cluster-health", "name": "Cluster Health", "description": "Check pod status, node health, resource usage"},
    {"id": "triage-alerts", "name": "Triage Alerts", "description": "Analyze and prioritize active alerts"},
    {"id": "suggest-actions", "name": "Suggest Actions", "description": "Recommend operational actions or runbooks"},
]

class OpsState(TypedDict):
    task: str
    cluster_status: dict
    alerts: list
    analysis: str
    recommendations: list[str]
    messages: Annotated[list, operator.add]

def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514"),
        openai_api_base=os.getenv("LITELLM_API_URL", "http://litellm.litellm.svc:4000"),
        openai_api_key=os.getenv("LITELLM_API_KEY", os.getenv("OPENAI_API_KEY", "sk-placeholder")),
    )

async def check_cluster(state: OpsState) -> OpsState:
    """Check Kubernetes cluster health."""
    status = {"pods": [], "nodes": [], "events": []}
    try:
        # When running in-cluster, uses service account
        result = subprocess.run(
            ["kubectl", "get", "pods", "--all-namespaces", "--field-selector=status.phase!=Running,status.phase!=Succeeded", "-o", "json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            status["unhealthy_pods"] = len(data.get("items", []))

        result2 = subprocess.run(
            ["kubectl", "get", "nodes", "-o", "json"],
            capture_output=True, text=True, timeout=30,
        )
        if result2.returncode == 0:
            import json
            nodes = json.loads(result2.stdout)
            status["nodes"] = [
                {"name": n["metadata"]["name"], "conditions": [c["type"] for c in n["status"].get("conditions", []) if c["status"] == "True"]}
                for n in nodes.get("items", [])
            ]
    except Exception as e:
        status["error"] = str(e)
    return {**state, "cluster_status": status,
            "messages": [HumanMessage(content=f"Cluster check complete")]}

async def check_alerts(state: OpsState) -> OpsState:
    """Fetch active alerts from Grafana/Mimir."""
    grafana_url = os.getenv("GRAFANA_URL", "https://grafana.o8s.ai")
    api_key = os.getenv("GRAFANA_API_KEY", "")
    alerts = []
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{grafana_url}/api/v1/provisioning/alert-rules",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code == 200:
                alerts = resp.json()
    except Exception as e:
        alerts = [{"error": str(e)}]
    return {**state, "alerts": alerts,
            "messages": [HumanMessage(content=f"Found {len(alerts)} alert rules")]}

async def analyze_and_recommend(state: OpsState) -> OpsState:
    """Use LLM to analyze cluster state and recommend actions."""
    llm = get_llm()
    prompt = f"""You are an SRE/Ops agent for the o8s Kubernetes platform.

Cluster Status: {state.get('cluster_status', {})}
Active Alerts: {state.get('alerts', [])}
Task: {state.get('task', 'General health check')}

The platform runs on bare-metal Kubernetes with:
- Rook-Ceph storage, MetalLB load balancer, Traefik ingress
- 94+ services managed by ArgoCD across 17 sync waves
- Observability: Mimir (metrics), Loki (logs), Tempo (traces), Grafana

Provide:
1. **Health Summary** — Overall cluster health assessment
2. **Issues Found** — List with severity (critical/warning/info)
3. **Recommendations** — Specific actions (reference runbooks in docs/runbooks/ if applicable)
4. **Priority** — What to fix first and why
"""
    response = await llm.ainvoke([
        SystemMessage(content="You are an experienced SRE. Be specific about Kubernetes resources and namespaces."),
        HumanMessage(content=prompt),
    ])
    return {**state, "analysis": response.content, "recommendations": ["See analysis"],
            "messages": [response]}

builder = StateGraph(OpsState)
builder.add_node("check_cluster", check_cluster)
builder.add_node("check_alerts", check_alerts)
builder.add_node("analyze_and_recommend", analyze_and_recommend)

builder.set_entry_point("check_cluster")
builder.add_edge("check_cluster", "check_alerts")
builder.add_edge("check_alerts", "analyze_and_recommend")
builder.add_edge("analyze_and_recommend", END)

graph = builder.compile()
