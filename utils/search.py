"""Native model-provider web search for the research agent.

Instead of a third-party search API (Tavily) wired through a custom `@tool`,
we use the model provider's own **server-side web search tool**. The provider
runs the search inside its own infrastructure during the model call, grounds
the answer in live results, and returns citations — so there's no separate
search SDK, API key, or fallback plumbing to maintain.

You pass the tool as a plain spec in the agent's `tools` list; the provider
recognizes it and handles the search loop itself:

    from utils.search import web_search
    from deepagents import create_deep_agent
    from utils.models import model

    agent = create_deep_agent(model=model, tools=[web_search()])

`web_search()` returns the spec matched to the active provider (Anthropic by
default). Keep it provider-agnostic by deriving the provider from the model,
or pass `provider=` explicitly.
"""

from __future__ import annotations

from typing import Optional


def web_search(
    *,
    provider: str = "anthropic",
    max_uses: int = 3,
    max_results: Optional[int] = None,
) -> dict:
    """Return the provider-native web search tool spec for the agent's `tools` list.

    The model provider executes the search server-side and grounds its answer in
    the results — there is no local tool body to run and no third-party API key.

    Args:
        provider: which provider's native search to use. ``"anthropic"`` (default)
            uses Anthropic's ``web_search_20250305`` server tool; ``"openai"`` uses
            OpenAI's ``web_search`` built-in tool. Keep this aligned with the model
            in ``utils/models.py``.
        max_uses: Anthropic only — cap on how many searches the model may run per
            turn (keeps a research turn from looping). Ignored by OpenAI.
        max_results: Anthropic only — optional cap on results per search.

    Returns:
        A tool spec dict to drop directly into ``tools=[...]``.
    """
    p = provider.lower()
    if p == "anthropic":
        spec: dict = {"type": "web_search_20250305", "name": "web_search"}
        if max_uses is not None:
            spec["max_uses"] = max_uses
        if max_results is not None:
            spec["max_results"] = max_results
        return spec
    if p == "openai":
        # OpenAI's Responses API exposes web search as a built-in tool.
        return {"type": "web_search"}
    raise ValueError(
        f"Unknown provider {provider!r}. Use 'anthropic' or 'openai', "
        "and keep it aligned with the model in utils/models.py."
    )
