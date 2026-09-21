"""Deployable financial research agent used by Module 2.

Demonstrates:
- AGENTS.md for agent identity and instructions
- Skills for on-demand capabilities (LinkedIn, Twitter)
- Native model-provider web search (no third-party search API)
- Research subagent for delegated work
- CompositeBackend: FilesystemBackend for skills/AGENTS.md, StoreBackend for /memories/
- Human-in-the-loop on file writes
"""

import os
from datetime import datetime

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend

from utils.models import model
from utils.search import web_search

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Provider-native web search: the model runs the search server-side and grounds
# its answer in live results. No third-party search SDK or key. See utils/search.py.
search = web_search(provider="anthropic")


research_subagent = {
    "name": "research-agent",
    "description": "Delegate research tasks. Give one topic at a time.",
    "system_prompt": f"""You are a financial research assistant. Today is {datetime.now().strftime('%Y-%m-%d')}.
Use web search to gather information from public sources (filings, press releases, market news).
Structure findings with clear headings and inline citations.
Limit to 3 searches.""",
    "tools": [search],
}


def backend_factory(rt):
    """FilesystemBackend for disk access, /memories/ routed to StoreBackend."""
    return CompositeBackend(
        default=FilesystemBackend(root_dir=AGENT_DIR, virtual_mode=True),
        routes={"/memories/": StoreBackend()},
    )


agent = create_deep_agent(
    model=model,
    tools=[search],
    system_prompt="You are an expert financial research analyst.",
    memory=["./AGENTS.md"],
    skills=["./skills/"],
    subagents=[research_subagent],
    backend=backend_factory,
    interrupt_on={"write_file": True, "edit_file": True},
)
