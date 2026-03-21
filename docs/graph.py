"""Documentation Agent — Keeps brain/ docs up to date as repos change."""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import subprocess
import os

DESCRIPTION = "Monitors repository changes and keeps the o8s brain documentation up to date"
SKILLS = [
    {"id": "scan-changes", "name": "Scan Changes", "description": "Detect recent changes across all o8s repositories"},
    {"id": "update-docs", "name": "Update Documentation", "description": "Update brain/ docs to reflect code changes"},
    {"id": "audit-docs", "name": "Audit Documentation", "description": "Check docs for accuracy and staleness"},
]

REPOS = [
    "agent-controller",
    "mission-control",
    "orbit",
    "o8s-cloner",
    "o8s-transfer-agent",
    "gitops",
    "branding",
    "orchestrator",
]

class DocsState(TypedDict):
    task: str
    repo_changes: dict
    stale_docs: list[str]
    updates_needed: list[dict]
    analysis: str
    messages: Annotated[list, operator.add]

def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "claude-sonnet-oauth"),
        openai_api_base=os.getenv("OPENAI_API_BASE", os.getenv("LITELLM_API_URL", "http://litellm.litellm.svc:4000")),
        openai_api_key=os.getenv("OPENAI_API_KEY", os.getenv("LITELLM_API_KEY", "sk-placeholder")),
    )

async def scan_repos(state: DocsState) -> DocsState:
    """Scan all repos for recent changes."""
    base_path = os.getenv("REPOS_PATH", "/Users/moprin/code/o8s")
    changes = {}
    for repo in REPOS:
        repo_path = os.path.join(base_path, repo)
        if not os.path.isdir(os.path.join(repo_path, ".git")):
            continue
        try:
            result = subprocess.run(
                ["git", "-C", repo_path, "log", "--oneline", "--since=7 days ago", "-20"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                changes[repo] = result.stdout.strip().split("\n")
        except Exception as e:
            changes[repo] = [f"Error: {e}"]
    return {**state, "repo_changes": changes,
            "messages": [HumanMessage(content=f"Scanned {len(changes)} repos with recent changes")]}

async def check_brain_docs(state: DocsState) -> DocsState:
    """Check existing brain docs for staleness."""
    brain_path = os.getenv("BRAIN_PATH", "/Users/moprin/code/o8s/brain")
    stale = []
    for f in os.listdir(brain_path):
        if f.endswith(".md") and f != "README.md":
            repo_name = f.replace(".md", "").replace("-", "_")
            # Check if the corresponding repo had changes
            for repo in state.get("repo_changes", {}):
                if repo.replace("-", "_") == repo_name or repo.replace("-", "") == repo_name.replace("_", ""):
                    stale.append(f)
                    break
    return {**state, "stale_docs": stale,
            "messages": [HumanMessage(content=f"Found {len(stale)} potentially stale docs")]}

async def analyze_updates(state: DocsState) -> DocsState:
    """Use LLM to determine what documentation updates are needed."""
    llm = get_llm()
    prompt = f"""You are a documentation maintenance agent for the o8s platform.

Recent repository changes (last 7 days):
{state.get('repo_changes', {})}

Potentially stale brain docs: {state.get('stale_docs', [])}

Task: {state.get('task', 'Audit documentation freshness')}

For each repo with changes:
1. Summarize what changed
2. Determine if the brain/ doc needs updating
3. List specific sections that need revision
4. Suggest the updates needed

Be specific about which doc file and which section needs updating.
"""
    response = await llm.ainvoke([
        SystemMessage(content="You are a technical documentation expert. Be precise about what needs updating."),
        HumanMessage(content=prompt),
    ])
    return {**state, "analysis": response.content, "updates_needed": [{"doc": d} for d in state.get("stale_docs", [])],
            "messages": [response]}

builder = StateGraph(DocsState)
builder.add_node("scan_repos", scan_repos)
builder.add_node("check_brain_docs", check_brain_docs)
builder.add_node("analyze_updates", analyze_updates)

builder.set_entry_point("scan_repos")
builder.add_edge("scan_repos", "check_brain_docs")
builder.add_edge("check_brain_docs", "analyze_updates")
builder.add_edge("analyze_updates", END)

graph = builder.compile()
