"""Query Argo Workflows for build pipeline status."""

import os
from typing import Any

import httpx
from langchain_core.tools import tool

ARGO_API_URL = os.environ.get(
    "ARGO_API_URL",
    "http://argo-workflows-server.argo-workflows.svc:2746",
)


@tool
async def query_argo(
    namespace: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fetch workflow status from Argo Workflows.

    Args:
        namespace: Kubernetes namespace to query. None = all namespaces.
        limit: Maximum workflows to return (default 50).

    Returns:
        List of workflow dicts: name, namespace, phase, started_at, finished_at,
        duration_seconds, message.
    """
    params: dict[str, Any] = {"listOptions.limit": limit}
    url = (
        f"{ARGO_API_URL}/api/v1/workflows/{namespace}"
        if namespace
        else f"{ARGO_API_URL}/api/v1/workflows"
    )

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    workflows = data.get("items") or []

    results = []
    for wf in workflows:
        meta = wf.get("metadata", {})
        status = wf.get("status", {})

        started = status.get("startedAt")
        finished = status.get("finishedAt")

        duration: float | None = None
        if started:
            from datetime import datetime, timezone

            start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            end_dt = (
                datetime.fromisoformat(finished.replace("Z", "+00:00"))
                if finished
                else datetime.now(tz=timezone.utc)
            )
            duration = (end_dt - start_dt).total_seconds()

        results.append(
            {
                "name": meta.get("name"),
                "namespace": meta.get("namespace"),
                "phase": status.get("phase", "Unknown"),
                "started_at": started,
                "finished_at": finished,
                "duration_seconds": duration,
                "message": status.get("message"),
                "labels": meta.get("labels", {}),
            }
        )

    return results
