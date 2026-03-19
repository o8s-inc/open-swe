"""Query the o8s tracker API for ticket information."""

import os
from typing import Any

import httpx
from langchain_core.tools import tool

TRACKER_API_URL = os.environ.get("TRACKER_API_URL", "http://localhost:3213")


@tool
async def query_tracker(
    status: str | None = None,
    assignee: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fetch tickets from the o8s tracker.

    Args:
        status: Filter by status (todo, in_progress, in_review, done). None = all.
        assignee: Filter by assignee ID. None = all.
        limit: Maximum number of tickets to return (default 50).

    Returns:
        List of ticket dicts with id, identifier, title, status, priority, assignee, labels.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if assignee:
            params["assigneeId"] = assignee

        resp = await client.get(
            f"{TRACKER_API_URL}/api/v1/issues",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

    # Normalise: support both array and {issues: [...]} shapes
    if isinstance(data, list):
        issues = data
    elif isinstance(data, dict):
        issues = data.get("issues", data.get("data", []))
    else:
        issues = []

    return [
        {
            "id": t.get("id"),
            "identifier": t.get("identifier"),
            "title": t.get("title"),
            "status": t.get("status"),
            "priority": t.get("priority"),
            "assignee": t.get("assignee") or t.get("assigneeId"),
            "labels": t.get("labels", []),
        }
        for t in issues[:limit]
    ]
