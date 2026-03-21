"""Code Review Agent — Reviews PRs using GitHub + Context7 for docs lookup."""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import httpx
import os

DESCRIPTION = "Reviews GitHub pull requests with documentation-aware analysis using Context7"
SKILLS = [
    {"id": "review-pr", "name": "Review PR", "description": "Analyze a GitHub PR for issues, style, and correctness"},
    {"id": "check-docs", "name": "Check Documentation", "description": "Verify code changes against library docs"},
    {"id": "suggest-improvements", "name": "Suggest Improvements", "description": "Propose code improvements"},
]

class ReviewState(TypedDict):
    repo: str
    pr_number: int
    diff: str
    files_changed: list[str]
    doc_context: str
    review: str
    comments: list[dict]
    messages: Annotated[list, operator.add]

def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "claude-sonnet-oauth"),
        openai_api_base=os.getenv("OPENAI_API_BASE", os.getenv("LITELLM_API_URL", "http://litellm.litellm.svc:4000")),
        openai_api_key=os.getenv("OPENAI_API_KEY", os.getenv("LITELLM_API_KEY", "sk-placeholder")),
    )

async def fetch_pr(state: ReviewState) -> ReviewState:
    """Fetch PR details from GitHub."""
    token = os.getenv("GITHUB_TOKEN", "")
    repo = state.get("repo", "")
    pr_number = state.get("pr_number", 0)
    diff = ""
    files_changed = []
    try:
        async with httpx.AsyncClient() as client:
            # Get PR diff
            resp = await client.get(
                f"https://api.github.com/repos/{repo}/pulls/{pr_number}",
                headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3.diff"},
            )
            if resp.status_code == 200:
                diff = resp.text[:50000]  # Cap at 50k chars
            # Get files changed
            resp2 = await client.get(
                f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp2.status_code == 200:
                files_changed = [f["filename"] for f in resp2.json()]
    except Exception as e:
        diff = f"Error fetching PR: {e}"
    return {**state, "diff": diff, "files_changed": files_changed,
            "messages": [HumanMessage(content=f"Fetched PR #{pr_number}: {len(files_changed)} files changed")]}

async def lookup_docs(state: ReviewState) -> ReviewState:
    """Look up relevant documentation via Context7 MCP."""
    # In production, this would call the Context7 MCP server
    # For now, we include file extension detection for doc context
    extensions = set()
    for f in state.get("files_changed", []):
        ext = f.rsplit(".", 1)[-1] if "." in f else ""
        extensions.add(ext)

    doc_hints = []
    ext_to_lib = {"tsx": "React", "ts": "TypeScript", "py": "Python", "go": "Go", "rs": "Rust", "yaml": "Kubernetes"}
    for ext in extensions:
        if ext in ext_to_lib:
            doc_hints.append(ext_to_lib[ext])

    doc_context = f"Libraries detected: {', '.join(doc_hints)}" if doc_hints else "No specific libraries detected"
    return {**state, "doc_context": doc_context,
            "messages": [HumanMessage(content=f"Doc context: {doc_context}")]}

async def review_code(state: ReviewState) -> ReviewState:
    """Use LLM to review the code changes."""
    llm = get_llm()
    prompt = f"""You are a senior code reviewer for the o8s platform.

Review this pull request:
- Repository: {state.get('repo')}
- PR #{state.get('pr_number')}
- Files changed: {', '.join(state.get('files_changed', []))}
- Documentation context: {state.get('doc_context', '')}

Diff:
```
{state.get('diff', '')[:30000]}
```

Provide:
1. **Summary** — What does this PR do?
2. **Issues** — Bugs, security issues, logic errors (with file:line references)
3. **Suggestions** — Code quality improvements
4. **Verdict** — APPROVE, REQUEST_CHANGES, or COMMENT
"""
    response = await llm.ainvoke([
        SystemMessage(content="You are a meticulous code reviewer. Be specific and reference line numbers."),
        HumanMessage(content=prompt),
    ])
    return {**state, "review": response.content, "messages": [response]}

async def post_review(state: ReviewState) -> ReviewState:
    """Post review comments back to GitHub."""
    comments = []
    if state.get("review"):
        comments.append({
            "action": "review_posted",
            "repo": state.get("repo"),
            "pr": state.get("pr_number"),
            "body": state.get("review", "")[:65000],
        })
    return {**state, "comments": comments}

builder = StateGraph(ReviewState)
builder.add_node("fetch_pr", fetch_pr)
builder.add_node("lookup_docs", lookup_docs)
builder.add_node("review_code", review_code)
builder.add_node("post_review", post_review)

builder.set_entry_point("fetch_pr")
builder.add_edge("fetch_pr", "lookup_docs")
builder.add_edge("lookup_docs", "review_code")
builder.add_edge("review_code", "post_review")
builder.add_edge("post_review", END)

graph = builder.compile()
