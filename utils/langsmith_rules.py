"""Helpers for managing LangSmith run rules from code.

LangSmith supports two related automations on a tracing project:

1. **Online evaluators** — score each new run with an LLM-as-judge and attach
   the score as feedback.
2. **Annotation queue automations** — route runs matching a filter into a
   queue for a human to review.

Both are configured as "rules" against a project. The LangSmith Python SDK
doesn't expose a high-level API for them, so we call the REST endpoint
directly. `create_run_rule` returns a deep link to the rule's page in the UI.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
from uuid import UUID

import requests
from langsmith import Client


# --------------------------------------------------------------------------- #
# Annotation queues
# --------------------------------------------------------------------------- #

def get_or_create_annotation_queue(
    client: Client,
    name: str,
    description: str = "",
):
    """Return an existing annotation queue by name, or create one."""
    existing = list(client.list_annotation_queues(name=name))
    if existing:
        return existing[0]
    return client.create_annotation_queue(name=name, description=description)


# --------------------------------------------------------------------------- #
# Run rules
# --------------------------------------------------------------------------- #

def _llm_judge_evaluator(
    prompt: Union[str, Sequence[tuple[str, str]]],
    output_schema: dict,
    *,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
    input_var: str = "input",
    output_var: str = "output",
) -> dict:
    """Build the `evaluators[]` payload for an LLM-as-judge online evaluator.

    `prompt` may be either:
      - a system prompt string (we'll add a default user template), or
      - a list of (role, content) tuples (full message list).

    `output_schema` is a JSON Schema dict with `title`, `description`,
    `properties`, and `required`. The LangSmith API requires those four fields.
    """
    if isinstance(prompt, str):
        messages = [
            ("system", prompt),
            ("human", f"Input: {{{{{input_var}}}}}\n\nOutput: {{{{{output_var}}}}}"),
        ]
    else:
        messages = list(prompt)

    return {
        "structured": {
            "prompt": [list(m) for m in messages],
            "model": {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
                "kwargs": {
                    "model": model_name,
                    "temperature": temperature,
                    # References the workspace-stored OPENAI_API_KEY secret —
                    # set one in LangSmith → Workspace settings → Secrets.
                    "api_key": {"lc": 1, "type": "secret", "id": ["OPENAI_API_KEY"]},
                },
            },
            "variable_mapping": {input_var: input_var, output_var: output_var},
            "schema": output_schema,
        }
    }


def create_run_rule(
    client: Client,
    *,
    project_name: str,
    display_name: str,
    filter: str = "",
    sampling_rate: float = 1.0,
    # If set: attach an LLM-as-judge online evaluator.
    llm_judge_prompt: Optional[Union[str, Sequence[tuple[str, str]]]] = None,
    llm_judge_schema: Optional[dict] = None,
    llm_judge_model: str = "gpt-4o-mini",
    # If set: route matching runs to this annotation queue.
    add_to_annotation_queue_id: Optional[Union[str, UUID]] = None,
) -> dict:
    """Create or replace a run rule on a tracing project.

    Returns a dict with `id`, `url` (deep link to the rule in the UI), and the
    raw `payload` LangSmith stored. Either `llm_judge_prompt` (+schema), or
    `add_to_annotation_queue_id`, or both, should be provided.
    """
    project = client.read_project(project_name=project_name)

    evaluators = []
    if llm_judge_prompt is not None:
        if llm_judge_schema is None:
            raise ValueError("llm_judge_schema is required when llm_judge_prompt is set")
        evaluators.append(
            _llm_judge_evaluator(
                llm_judge_prompt, llm_judge_schema, model_name=llm_judge_model,
            )
        )

    body = {
        "display_name": display_name,
        "session_id": str(project.id),
        "sampling_rate": sampling_rate,
        "filter": filter,
        "evaluators": evaluators,
    }
    if add_to_annotation_queue_id is not None:
        body["add_to_annotation_queue_id"] = str(add_to_annotation_queue_id)

    headers = {
        "x-api-key": client.api_key,
        "content-type": "application/json",
        "accept": "application/json",
    }

    # Idempotency: if a rule with this display_name already exists in the project,
    # delete it first. POST /runs/rules has no upsert semantics, so without this
    # rerunning a notebook cell accumulates duplicate rules.
    list_response = requests.get(
        f"{client.api_url}/runs/rules",
        params={"session_id": str(project.id)},
        headers={"x-api-key": client.api_key, "accept": "application/json"},
        timeout=30,
    )
    list_response.raise_for_status()
    for existing in list_response.json():
        if existing.get("display_name") == display_name:
            requests.delete(
                f"{client.api_url}/runs/rules/{existing['id']}",
                headers={"x-api-key": client.api_key, "accept": "application/json"},
                timeout=15,
            )

    response = requests.post(
        f"{client.api_url}/runs/rules", json=body, headers=headers, timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    tenant_id = payload["tenant_id"]
    rule_id = payload["id"]
    evaluator_id = payload.get("evaluator_id")

    # Evaluator rules (LLM-as-judge) and automation rules (queue routing, etc.)
    # have different UI pages in LangSmith.
    if evaluator_id:
        url = (
            f"https://smith.langchain.com/o/{tenant_id}/evaluators/{evaluator_id}"
            f"?ruleId={rule_id}&sourceKind=session&sourceId={project.id}"
        )
    else:
        url = (
            f"https://smith.langchain.com/o/{tenant_id}/projects/p/{project.id}"
            f"?runview=threads&tab=2"
        )

    return {"id": rule_id, "url": url, "payload": payload}


def delete_run_rule(client: Client, rule_id: Union[str, UUID]) -> None:
    """Delete a run rule by id."""
    headers = {"x-api-key": client.api_key, "accept": "application/json"}
    response = requests.delete(
        f"{client.api_url}/runs/rules/{rule_id}", headers=headers, timeout=15,
    )
    response.raise_for_status()
