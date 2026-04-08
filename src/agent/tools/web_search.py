from __future__ import annotations

import os

from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_tavily import TavilySearchResults


def _is_tavily_low_usage_error(err: BaseException) -> bool:
    """
    Best-effort detection of "free usage is low / quota exceeded" failures.
    Tavily errors can vary by transport/version, so we match common signals.
    """
    msg = str(err).lower()
    return any(
        s in msg
        for s in (
            "quota",
            "rate limit",
            "rate-limit",
            "too many requests",
            "insufficient credits",
            "insufficient credit",
            "payment required",
            "plan",
            "402",
            "429",
        )
    )


def _get_ddg_tool() -> DuckDuckGoSearchResults:
    return DuckDuckGoSearchResults()


def _get_tavily_tool() -> TavilySearchResults | None:
    # LangChain's Tavily tool reads TAVILY_API_KEY from the environment.
    if not os.getenv("TAVILY_API_KEY"):
        return None
    return TavilySearchResults()


@tool
def web_search(query: str) -> str:
    """
    Web search with Tavily-first fallback to DuckDuckGo.

    Behavior:
    - If `TAVILY_API_KEY` is set, try Tavily first.
    - If Tavily fails due to "low free usage" / quota / rate limiting (or returns
      an empty result), fall back to DuckDuckGoSearchResults.
    - If `TAVILY_API_KEY` is not set, use DuckDuckGoSearchResults directly.
    """
    tavily = _get_tavily_tool()
    ddg = _get_ddg_tool()

    if tavily is None:
        return ddg.run(query)

    try:
        result = tavily.run(query)
        if result is None or (isinstance(result, str) and not result.strip()):
            return ddg.run(query)
        return result
    except Exception as e:
        if _is_tavily_low_usage_error(e):
            return ddg.run(query)
        raise