"""Shared financial research agent used by Module 1 (Deep Agents) and Module 3 (LangSmith).

This is the minimal useful Deep Agent: a financial research subagent + the model
provider's native web search + a checkpointer. It deliberately omits HITL and
FilesystemBackend so that evaluation runs in Module 3 don't pause or leak files
to disk.

Module 1 builds up to this agent step-by-step in the notebook. This file
packages the same pattern so Module 3 can import it directly:

    from agents.research_agent import build_research_agent
    agent = build_research_agent()

Module 3 also exercises two optional knobs:

- ``model=`` — swap in a different chat model (the model-comparison experiment
  runs the same agent on a larger vs. a smaller model).
- ``context_repo=`` — mount the agent's operating manual (``AGENTS.md``) and
  skills from a versioned **LangSmith Context Hub** repo instead of local disk.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

from utils.models import model as default_model
from utils.search import web_search


def _materialize_context_repo(context_repo: str) -> Path:
    """Pull a Context Hub repo and write its files to a temp dir.

    Returns the directory containing ``AGENTS.md`` and a ``skills/`` tree, so it
    can be passed to ``create_deep_agent(memory=..., skills=...)``. The repo is
    the same source of truth used elsewhere; pulling it here means the manual and
    skills are versioned in the Hub rather than pinned to a local checkout.
    """
    from langsmith import Client

    ctx = Client().pull_agent(context_repo)
    root = Path(tempfile.mkdtemp(prefix="research-context-"))
    for path, entry in ctx.files.items():
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(entry.content, encoding="utf-8")
    return root


def build_research_agent(
    *,
    model=None,
    context_repo: Optional[str] = None,
):
    """Return a fresh financial research deep agent.

    Each call returns a new agent with a fresh checkpointer — useful so eval
    runs don't share state with each other.

    Args:
        model: chat model to power the agent. Defaults to the shared workshop
            model in ``utils/models.py``. Module 3's model-comparison experiment
            passes a larger vs. a smaller model here.
        context_repo: optional LangSmith Context Hub repo identifier. When set,
            the agent's ``AGENTS.md`` operating manual and ``skills/`` are pulled
            from the Hub and mounted, instead of using the built-in prompt.
    """
    model = model or default_model

    # Provider-native web search: the model runs the search server-side and
    # grounds its answer in live results. No third-party search SDK or key.
    search = web_search(provider="anthropic")

    research_subagent = {
        "name": "research-agent",
        "description": "Delegate research tasks. Give one topic at a time.",
        "system_prompt": (
            f"You are a financial research assistant. Today is "
            f"{datetime.now().strftime('%Y-%m-%d')}.\n"
            "Use web search to gather information from public sources. "
            "Limit to 3 searches."
        ),
        "tools": [search],
    }

    kwargs = dict(
        model=model,
        tools=[search],
        subagents=[research_subagent],
        checkpointer=MemorySaver(),
    )

    if context_repo:
        # Mount the manual + skills from the Hub repo.
        ctx_dir = _materialize_context_repo(context_repo)
        kwargs["memory"] = [str(ctx_dir / "AGENTS.md")]
        kwargs["skills"] = [str(ctx_dir / "skills") + "/"]
        kwargs["system_prompt"] = "You are an expert financial research analyst."
    else:
        kwargs["system_prompt"] = (
            "You are a helpful financial research assistant. "
            "Delegate research to the research-agent."
        )

    return create_deep_agent(**kwargs)
