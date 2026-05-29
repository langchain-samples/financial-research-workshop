"""Shared financial research agent used by Module 1 (Deep Agents) and Module 3 (LangSmith).

This is the minimal useful Deep Agent: a financial research subagent + Tavily search + a
checkpointer. It deliberately omits HITL and FilesystemBackend so that
evaluation runs in Module 3 don't pause or leak files to disk.

Module 1 builds up to this agent step-by-step in the notebook. This file
packages the same pattern so Module 3 can import it directly:

    from agents.research_agent import build_research_agent
    agent = build_research_agent()
"""

from datetime import datetime

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from utils.models import model
from utils.search import resilient_tavily_search


@tool(parse_docstring=True)
def tavily_search(query: str) -> str:
    """Search the web for information on a given query.

    Args:
        query: Search query to execute.
    """
    # `resilient_tavily_search` retries on Tavily failure with linear backoff,
    # then falls back to a topic-matched canned response so a flaky network
    # doesn't break the workshop. See utils/search.py for details.
    return resilient_tavily_search(query, max_retries=2)


def build_research_agent():
    """Return a fresh research deep agent.

    Each call returns a new agent with a fresh checkpointer — useful so eval
    runs don't share state with each other.
    """
    research_subagent = {
        "name": "research-agent",
        "description": "Delegate research tasks. Give one topic at a time.",
        "system_prompt": (
            f"You are a financial research assistant. Today is "
            f"{datetime.now().strftime('%Y-%m-%d')}.\n"
            "Use tools to gather information from public sources. Limit to 3 search calls."
        ),
        "tools": [tavily_search],
    }

    return create_deep_agent(
        model=model,
        tools=[tavily_search],
        system_prompt=(
            "You are a helpful financial research assistant. "
            "Delegate research to the research-agent."
        ),
        subagents=[research_subagent],
        checkpointer=MemorySaver(),
    )
