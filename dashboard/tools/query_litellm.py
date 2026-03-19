"""Query the LiteLLM spend API for cost information."""

import os
from typing import Any

import httpx
from langchain_core.tools import tool

LITELLM_API_URL = os.environ.get("LITELLM_API_URL", "http://litellm.litellm.svc:4000")


@tool
async def query_litellm(
    period: str = "month",
) -> dict[str, Any]:
    """Fetch LLM spend/cost data from LiteLLM.

    Args:
        period: Time period — 'today', 'week', or 'month'.

    Returns:
        Dict with total_spend, by_model (list), and by_user (list).
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{LITELLM_API_URL}/spend/logs")
        resp.raise_for_status()
        data = resp.json()

    records: list[dict[str, Any]] = (
        data if isinstance(data, list)
        else data.get("response", data.get("data", []))
    )

    from datetime import datetime, timedelta, timezone

    now = datetime.now(tz=timezone.utc)
    cutoffs = {
        "today": now.replace(hour=0, minute=0, second=0, microsecond=0),
        "week": now - timedelta(days=7),
        "month": now - timedelta(days=30),
    }
    cutoff = cutoffs.get(period, cutoffs["month"])

    total = 0.0
    by_model: dict[str, float] = {}
    by_user: dict[str, float] = {}

    for r in records:
        spend = float(r.get("spend") or 0)
        ts_str = r.get("startTime") or r.get("created_at") or ""
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts = datetime.min.replace(tzinfo=timezone.utc)

        if ts < cutoff:
            continue

        total += spend
        model = r.get("model", "unknown")
        user = r.get("user") or r.get("api_key") or "unknown"
        by_model[model] = by_model.get(model, 0.0) + spend
        by_user[user] = by_user.get(user, 0.0) + spend

    return {
        "period": period,
        "total_spend": round(total, 6),
        "currency": "USD",
        "by_model": sorted(
            [{"model": k, "spend": round(v, 6)} for k, v in by_model.items()],
            key=lambda x: x["spend"],
            reverse=True,
        ),
        "by_user": sorted(
            [{"user": k, "spend": round(v, 6)} for k, v in by_user.items()],
            key=lambda x: x["spend"],
            reverse=True,
        ),
    }
