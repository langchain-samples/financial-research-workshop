"""Deployable financial research agent used by Module 2.

Demonstrates:
- AGENTS.md for agent identity and instructions
- Skills for on-demand capabilities (LinkedIn, Twitter)
- Custom tools (Tavily search)
- Research subagent for delegated work
- CompositeBackend: FilesystemBackend for skills/AGENTS.md, StoreBackend for /memories/
- Human-in-the-loop on file writes
"""

import os
from datetime import datetime

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langchain_core.tools import tool

from utils.models import model
from utils.search import resilient_tavily_search

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


@tool(parse_docstring=True)
def tavily_search(query: str) -> str:
    """Search the web for information on a given query.

    Args:
        query: Search query to execute.
    """
    # Resilient wrapper: retries on Tavily failure, then falls back to a
    # topic-matched canned response. See utils/search.py.
    return resilient_tavily_search(query, max_retries=2)


research_subagent = {
    "name": "research-agent",
    "description": "Delegate research tasks. Give one topic at a time.",
    "system_prompt": f"""You are a financial research assistant. Today is {datetime.now().strftime('%Y-%m-%d')}.
Use tools to gather information from public sources (filings, press releases, market news).
Structure findings with clear headings and inline citations.
Limit to 3 search calls.""",
    "tools": [tavily_search],
}


def backend_factory(rt):
    """FilesystemBackend for disk access, /memories/ routed to StoreBackend."""
    return CompositeBackend(
        default=FilesystemBackend(root_dir=AGENT_DIR, virtual_mode=True),
        routes={"/memories/": StoreBackend()},
    )


agent = create_deep_agent(
    model=model,
    tools=[tavily_search],
    system_prompt="You are an expert financial research analyst.",
    memory=["./AGENTS.md"],
    skills=["./skills/"],
    subagents=[research_subagent],
    backend=backend_factory,
    interrupt_on={"write_file": True, "edit_file": True},
)
