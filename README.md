# Finance Research Agent Workshop

A guided tour of the LangChain platform built around a single use case: a financial research agent that goes from sandbox to a governed production deployment with full observability. Four standalone Jupyter modules that combine into ~1.5 to 2 hour workshops.

The agent itself is a generalist financial research analyst — it researches equities, fixed income, macro, and sectors from public sources, then writes investor notes or earnings summaries. The same agent serves as the running example across every module.

## The Modules

| # | Module | Duration | Notebook |
|---|--------|----------|----------|
| **1** | Deep Agents — harness, custom tools, subagents, memory, middleware, HITL, AGENTS.md and Skills | ~45 min | `modules/01_deep_agents.ipynb` |
| **2** | Move to Production — workspace policies via the LangSmith LLM Gateway, then deploy with `langgraph` CLI to LangSmith Deployments | ~25 min | `modules/02_deploy_and_govern.ipynb` |
| **3** | LangSmith — tracing and querying traces, Engine for automated failure-mode discovery, offline + online evaluations, annotation queues | ~30 min | `modules/03_langsmith.ipynb` |
| **4** | Accelerate LangGraph with NVIDIA — parallel and speculative execution via `langchain-nvidia-langgraph` | ~10 min | `modules/04_nvidia.ipynb` |

Each module is a standalone Jupyter notebook. Modules share the project's setup, `utils/`, and `agents/`, so combining them is as straightforward as opening multiple notebooks in order.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment variables
cp .env.example .env
# Edit .env and fill in your keys
```

| Key | Required for | Notes |
|-----|--------------|-------|
| `LANGSMITH_API_KEY` | All modules | Tracing, deployments, gateway, evaluations. Use a service key (`lsv2_sk_...`) for Module 2 deploys. |
| `ANTHROPIC_API_KEY` | All modules | Default model (`claude-sonnet-4-6`) is routed through the LangSmith LLM Gateway. |
| `WORKSPACE_ID` | Module 2 | The LangSmith workspace the gateway policy applies to. Find it in Settings → Workspace. |
| `LANGSMITH_API_KEY_GATEWAY` | Module 2 (after §1.4 flip) and the deployed agent | Same value as `LANGSMITH_API_KEY`. Required under a non-reserved name because `langgraph deploy` strips `LANGSMITH_API_KEY` from deployed containers. |
| `TAVILY_API_KEY` | Modules 1 and 3 | Web search tool used by the research agent. <https://tavily.com> |
| `OPENAI_API_KEY` | Optional | Only required if you swap the default model in `utils/models.py` to OpenAI. |

```bash
# 3. Start Jupyter
uv run jupyter notebook
```

Open whichever module(s) you intend to run.

## Default Model and the LangSmith LLM Gateway

The workshop's default model is configured in `utils/models.py`:

```python
model = init_chat_model(
    model="claude-sonnet-4-6",
    model_provider="anthropic",
    base_url="https://gateway.smith.langchain.com/anthropic",
)
```

The `base_url` routes every model call through the **LangSmith LLM Gateway**, which means any workspace-level policy (PII detection, secrets redaction, allow-lists, rate limits, cost caps) applies uniformly across the workshop — including to the deployable agent. Module 2 walks through configuring a sample policy.

To switch providers, edit `utils/models.py`:

```python
# Anthropic, direct (bypasses the gateway and any workspace policies)
# model = init_chat_model("anthropic:claude-sonnet-4-6")

# OpenAI
# model = init_chat_model("openai:gpt-4.1-mini")

# Azure OpenAI
# from langchain_openai import AzureChatOpenAI
# model = AzureChatOpenAI(azure_deployment="gpt-4.1-mini", streaming=True)

# AWS Bedrock
# from langchain_aws import ChatBedrockConverse
# model = ChatBedrockConverse(provider="anthropic", model_id="...")
```

## Deploy (Module 2)

Module 2 deploys the agent at `agents/deep_agent/` to LangSmith via the `langgraph` CLI (installed by `uv sync`). The deploy config is `langgraph.json` at the workshop root. The deployment name is `financial-research-agent`.

The `LANGSMITH_API_KEY` used for deploys must have deployment permissions (a `lsv2_sk_...` service key, not a personal token).

## Project Structure

```
finance-research-agent-workshop/
├── README.md                       (this file — setup + reference)
├── pyproject.toml                  (shared dependencies)
├── .env.example
├── langgraph.json                  (registers agents/deep_agent for langgraph dev)
├── utils/
│   ├── models.py                   (default model, gateway-routed)
│   ├── search.py                   (Tavily wrapper with topic-matched fallbacks)
│   └── langsmith_rules.py          (run rule + annotation queue helpers)
├── agents/
│   ├── research_agent.py           (shared agent factory — Module 1 references, Module 3 imports for eval)
│   └── deep_agent/                 (deployable agent for Module 2)
│       ├── agent.py
│       ├── AGENTS.md               (agent identity)
│       ├── deepagents.toml
│       └── skills/
│           ├── investor-note/SKILL.md
│           └── earnings-summary/SKILL.md
├── images/                         (diagrams + screenshots used by the notebooks)
└── modules/
    ├── 01_deep_agents.ipynb
    ├── 02_deploy_and_govern.ipynb
    ├── 03_langsmith.ipynb
    └── 04_nvidia.ipynb
```

## Common Issues

**`langgraph deploy` fails with 403 / permission denied**
The API key in use is a personal token. Generate a service key (`lsv2_sk_...`) in LangSmith settings.

**Module 2 policy creation fails with 401 / unauthorized**
`LANGSMITH_API_KEY` lacks the workspace admin scope required to create gateway policies, or `WORKSPACE_ID` does not match the workspace the key belongs to.

**Notebook can't find `utils` / `agents`**
Each module's setup cell prepends the workshop root to `sys.path`. If a notebook has been moved, update the `Path().resolve().parent` line to point at the workshop root.
