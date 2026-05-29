"""Retry + canned-fallback helper for Tavily web search.

The shape: call `resilient_tavily_search(query)` from inside your own `@tool`
function. The helper retries on Tavily failures with linear backoff, and if
every retry fails it returns a Tavily-shaped canned response matched against
topic keywords — chosen so the agent's downstream synthesis still produces a
useful answer during a workshop with a flaky network.

Usage (from a notebook or module):

    from langchain_core.tools import tool
    from utils.search import resilient_tavily_search

    @tool(parse_docstring=True)
    def tavily_search(query: str) -> str:
        \"\"\"Search the web for information on a given query.

        Args:
            query: Search query to execute.
        \"\"\"
        return resilient_tavily_search(query, max_retries=2)

The `@tool` decorator stays at the call site so the tool's name, docstring,
and signature remain visible to the reader. Only the resilience plumbing is
hidden behind the util.
"""

from __future__ import annotations

import time
from typing import Optional

from tavily import TavilyClient


# --------------------------------------------------------------------------- #
# Canned fallbacks — based on patterns we observed in real traces.
# Each entry is a list of (title, url, content) tuples shaped like Tavily's
# `results[]` payload. Keys are lowercase phrases matched as substrings.
# Order matters: longer/more-specific keys are checked first.
# --------------------------------------------------------------------------- #

_FALLBACK_RESULTS: list[tuple[tuple[str, ...], list[tuple[str, str, str]]]] = [
    (("apple earnings", "apple inc", "aapl", "apple quarterly", "apple q"), [
        (
            "Apple — recent quarterly results",
            "https://www.apple.com/newsroom/",
            "Apple's most recent quarter showed Services revenue continuing to outpace overall growth, "
            "with iPhone unit shipments roughly stable year-over-year. Operating margin expanded modestly "
            "on a richer Services mix. Management reiterated capital-return commitments and pointed to a "
            "steady demand backdrop in mature markets while flagging FX headwinds.",
        ),
    ]),
    (("federal funds rate", "fed funds", "fomc", "fed rate", "rate decision", "federal reserve"), [
        (
            "FOMC — most recent policy statement",
            "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            "The Federal Open Market Committee held the federal funds target range at the most recent "
            "meeting, citing inflation that remains above the 2 percent objective alongside a labor market "
            "that has come into better balance. The statement reiterated a data-dependent approach and "
            "left forward guidance largely unchanged.",
        ),
    ]),
    (("us cpi", "consumer price index", "inflation report", "cpi release"), [
        (
            "US CPI — latest release",
            "https://www.bls.gov/cpi/",
            "Headline CPI rose at an annualized pace consistent with the gradual disinflation trend "
            "observed over the last year. Core CPI (ex food and energy) was somewhat stickier, driven "
            "by services categories including shelter and insurance. Goods inflation remained subdued.",
        ),
    ]),
    (("vix", "implied volatility", "volatility index"), [
        (
            "VIX — what it measures",
            "https://www.cboe.com/tradable_products/vix/",
            "The CBOE Volatility Index (VIX) is a real-time index representing the market's expectation "
            "of 30-day forward-looking volatility, derived from S&P 500 index option prices. It is "
            "commonly cited as a gauge of market sentiment and is sometimes referred to as the 'fear index'.",
        ),
    ]),
    (("high-yield bond", "high yield bond", "junk bond", "credit spread"), [
        (
            "High-yield bonds — overview",
            "https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/high-yield",
            "High-yield (or 'junk') bonds are corporate debt instruments rated below investment grade "
            "by major credit rating agencies. They offer higher coupons to compensate for higher default "
            "risk. Credit spreads — the yield premium over comparable Treasuries — widen when investors "
            "demand additional compensation for credit risk and narrow in risk-on environments.",
        ),
    ]),
    (("difference between langchain and langgraph", "langchain vs langgraph"), [
        (
            "LangChain vs LangGraph — when to use each",
            "https://docs.langchain.com/oss/python/concepts/products",
            "LangChain provides the high-level agent and chain abstractions (create_agent, "
            "tools, middleware) for assembling LLM applications. LangGraph is the lower-level "
            "runtime that LangChain agents compile to — a stateful graph with nodes, edges, "
            "checkpointers, and human-in-the-loop interrupts. Use LangChain when you want a "
            "ReAct-style agent in a few lines; use LangGraph when you need custom state, "
            "multi-agent topologies, or fine-grained control of the execution loop.",
        ),
    ]),
    (("create_agent", "create agent"), [
        (
            "create_agent — LangChain v1 prebuilt agent",
            "https://docs.langchain.com/oss/python/langchain/agents",
            "`create_agent(model, tools, system_prompt, middleware=...)` returns a ReAct-style "
            "agent as a compiled LangGraph. It handles the model→tool→model loop, accepts "
            "middleware for cross-cutting concerns (logging, HITL, structured output), and "
            "can be passed a checkpointer for persistence. Returned object exposes "
            ".invoke / .stream / .ainvoke and is itself a Runnable.",
        ),
    ]),
    (("langchain v1", "langchain 1.0"), [
        (
            "LangChain v1.0 release notes",
            "https://blog.langchain.com/langchain-v1/",
            "LangChain v1 introduces a redesigned agent API centered on `create_agent` and a "
            "first-class middleware system for customizing the agent loop. Other highlights: "
            "consolidated message types in langchain-core, an init_chat_model factory for "
            "provider-agnostic model loading, and improved structured output support via "
            "with_structured_output. Built on top of LangGraph for the runtime."
        ),
        (
            "Migrating to LangChain v1",
            "https://docs.langchain.com/oss/python/releases/langchain-v1",
            "v1 simplifies the agent surface area: AgentExecutor and the v0 agent constructors "
            "are replaced by `create_agent`, middleware replaces callbacks for most flows, and "
            "tool definitions remain compatible with the existing `@tool` decorator."
        ),
    ]),
    (("langgraph",), [
        (
            "LangGraph — stateful, multi-agent workflows",
            "https://docs.langchain.com/oss/python/langgraph/overview",
            "LangGraph is an open-source framework for building stateful, multi-agent "
            "applications with LLMs. You model your workflow as a StateGraph: define a "
            "TypedDict State, register nodes (Python functions) that read state and return "
            "updates, and connect them with normal or conditional edges. Built-in support for "
            "checkpointers (persistence), interrupts (human-in-the-loop), and stores "
            "(long-term memory)."
        ),
    ]),
    (("langsmith",), [
        (
            "LangSmith — observability and evaluation for LLM apps",
            "https://docs.smith.langchain.com/",
            "LangSmith is a framework-agnostic platform for tracing, evaluating, and monitoring "
            "LLM applications. Set LANGSMITH_TRACING=true and every LLM/tool/chain call lands "
            "in your project. Build datasets, run offline evaluations with `client.evaluate`, "
            "or configure online evaluators (run rules) that score new traces automatically. "
            "Annotation queues route flagged runs to human reviewers."
        ),
    ]),
    (("tavily",), [
        (
            "Tavily — search API for AI agents",
            "https://tavily.com/",
            "Tavily is a web search API optimized for AI agents and RAG pipelines. It returns "
            "ranked search results with extracted content blocks, designed to plug directly "
            "into LLM prompts. Has a Python SDK (`tavily-python`) with a simple "
            "TavilyClient.search(query, max_results=N) interface."
        ),
    ]),
]


_GENERIC_FALLBACK = [
    (
        "Search temporarily unavailable",
        "https://fallback.example/no-results",
        "The web search service didn't return results for this query. Use what you already "
        "know to answer; if information is missing, say so plainly rather than guessing.",
    ),
]


def _format_results(results: list[tuple[str, str, str]]) -> str:
    """Match TavilyClient output formatting used everywhere else in the workshops."""
    return "\n\n".join(f"**{t}**\n{u}\n{c}" for t, u, c in results)


def _pick_fallback(query: str) -> str:
    q = query.lower()
    for keys, results in _FALLBACK_RESULTS:
        if any(k in q for k in keys):
            return _format_results(results)
    return _format_results(_GENERIC_FALLBACK)


# --------------------------------------------------------------------------- #
# Public helper
# --------------------------------------------------------------------------- #

# Lazy singleton — instantiated once on first call, then reused.
_default_client: Optional[TavilyClient] = None


def resilient_tavily_search(
    query: str,
    *,
    max_retries: int = 2,
    max_results: int = 3,
    base_backoff_seconds: float = 1.0,
    client: Optional[TavilyClient] = None,
) -> str:
    """Run a Tavily search with retries; fall back to canned content on failure.

    Returns a Tavily-shaped string (each result formatted as
    `**title**\\nurl\\ncontent`, joined by blank lines) so the caller can return
    it from a `@tool` function as-is.

    Args:
        query: search query to run.
        max_retries: additional attempts after the first call. `max_retries=2`
            means up to 3 total attempts before falling back.
        max_results: passed through to `TavilyClient.search`.
        base_backoff_seconds: first retry sleeps this long; subsequent retries
            sleep `base * attempt` (linear backoff).
        client: optional TavilyClient to use. If `None`, a lazily-initialized
            module-level client is used.
    """
    global _default_client
    if client is None:
        if _default_client is None:
            _default_client = TavilyClient()
        client = _default_client

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            results = client.search(query, max_results=max_results)
            hits = results.get("results", [])
            if hits:
                return "\n\n".join(
                    f"**{r['title']}**\n{r['url']}\n{r['content']}"
                    for r in hits
                )
            last_error = RuntimeError("Tavily returned 0 results")
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(base_backoff_seconds * (attempt + 1))

    notice = (
        f"[fallback content; live search unavailable -- "
        f"{type(last_error).__name__}: {str(last_error)[:120]}]\n\n"
    )
    return notice + _pick_fallback(query)
