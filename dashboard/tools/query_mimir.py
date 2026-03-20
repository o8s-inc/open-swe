"""Query Mimir/Prometheus for cluster metrics."""

import os
from typing import Any

import httpx
from langchain_core.tools import tool

MIMIR_API_URL = os.environ.get("MIMIR_API_URL", "http://mimir-gateway.monitoring.svc:80")
MIMIR_ORG_ID = os.environ.get("MIMIR_ORG_ID", "mgmt")


async def _promql(query: str) -> list[dict[str, Any]]:
    """Execute a PromQL instant query against Mimir."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{MIMIR_API_URL}/prometheus/api/v1/query",
            params={"query": query},
            headers={"X-Scope-OrgID": MIMIR_ORG_ID},
        )
        resp.raise_for_status()
        data = resp.json()

    if data.get("status") != "success":
        raise RuntimeError(f"Mimir error: {data.get('error', 'unknown')}")

    return data.get("data", {}).get("result", [])


@tool
async def query_mimir(
    queries: list[str] | None = None,
) -> dict[str, Any]:
    """Query Mimir/Prometheus for cluster health metrics.

    Fetches: node count, pod count (total and running), CPU usage, memory usage,
    top 10 pods by memory, and pod counts per namespace.

    Args:
        queries: Optional list of extra PromQL queries to run.
                 Each item should be the raw PromQL string.

    Returns:
        Dict with node_count, pod_count, running_pods, cpu_cores_used,
        total_memory_gb, used_memory_gb, memory_pct, top_pods,
        pods_by_namespace, and optional extra_results.
    """
    import asyncio

    preset_queries = {
        "total_memory": "sum(node_memory_MemTotal_bytes)",
        "available_memory": "sum(node_memory_MemAvailable_bytes)",
        "cpu_usage": "sum(rate(node_cpu_seconds_total{mode!=\"idle\"}[5m]))",
        "node_count": "count(kube_node_info)",
        "pod_count": "count(kube_pod_info)",
        "running_pods": 'count(kube_pod_status_phase{phase="Running"})',
        "top_pods": (
            "topk(10, sum by (namespace,pod)"
            " (container_memory_working_set_bytes{container!=\"\"}))"
        ),
        "pods_by_namespace": "count by (namespace) (kube_pod_info)",
    }

    results = await asyncio.gather(
        *[_promql(q) for q in preset_queries.values()],
        return_exceptions=True,
    )

    def scalar(r: Any) -> float:
        if isinstance(r, Exception) or not r:
            return 0.0
        return float(r[0]["value"][1]) if r else 0.0

    total_mem, avail_mem, cpu, nodes, pod_count, running_pods, top_pods_raw, pods_by_ns_raw = results

    used_mem = scalar(total_mem) - scalar(avail_mem)
    total_mem_gb = scalar(total_mem) / (1024 ** 3)
    used_mem_gb = used_mem / (1024 ** 3)
    mem_pct = (used_mem / scalar(total_mem) * 100) if scalar(total_mem) > 0 else 0.0

    top_pods = []
    if not isinstance(top_pods_raw, Exception):
        for r in top_pods_raw:  # type: ignore[union-attr]
            top_pods.append({
                "namespace": r["metric"].get("namespace", ""),
                "pod": r["metric"].get("pod", ""),
                "memory_mb": round(float(r["value"][1]) / (1024 ** 2), 1),
            })

    pods_by_namespace = []
    if not isinstance(pods_by_ns_raw, Exception):
        for r in pods_by_ns_raw:
            pods_by_namespace.append({
                "namespace": r["metric"].get("namespace", ""),
                "count": int(float(r["value"][1])),
            })
        pods_by_namespace.sort(key=lambda x: x["count"], reverse=True)

    result: dict[str, Any] = {
        "node_count": int(scalar(nodes)),
        "pod_count": int(scalar(pod_count)),
        "running_pods": int(scalar(running_pods)),
        "cpu_cores_used": round(scalar(cpu), 2),
        "total_memory_gb": round(total_mem_gb, 1),
        "used_memory_gb": round(used_mem_gb, 1),
        "memory_pct": round(mem_pct, 1),
        "top_pods": top_pods,
        "pods_by_namespace": pods_by_namespace,
    }

    # Run extra queries if provided
    if queries:
        extra_results = await asyncio.gather(
            *[_promql(q) for q in queries],
            return_exceptions=True,
        )
        result["extra_results"] = [
            {"query": q, "result": r if not isinstance(r, Exception) else str(r)}
            for q, r in zip(queries, extra_results)
        ]

    return result
