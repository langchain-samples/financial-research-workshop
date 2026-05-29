"""Centralized model initialization.

The notebooks all import `model` from here, so swapping providers
(OpenAI / Anthropic / Azure / Bedrock) only requires editing this file.

The default is Anthropic, direct. Module 2 walks through introducing the
LangSmith LLM Gateway as the production-ready alternative — to make the
switch, comment out the default below and uncomment the gateway block.
"""

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env", override=True)

from langchain.chat_models import init_chat_model

# --- Default: Anthropic, direct ---
model = init_chat_model("anthropic:claude-sonnet-4-6")

# --- Anthropic via the LangSmith LLM Gateway (Module 2 §1.4) ---
# Routes every model call through the LangSmith Gateway so that workspace
# policies (PII / secrets / allow-lists / cost caps) are enforced.
# model = init_chat_model(
#     model="claude-sonnet-4-6",
#     model_provider="anthropic",
#     base_url="https://gateway.smith.langchain.com/anthropic",
#     api_key=os.environ["LANGSMITH_API_KEY_GATEWAY"],
# )

# --- OpenAI ---
# model = init_chat_model("openai:gpt-4.1-mini")

# --- Azure OpenAI ---
# from langchain_openai import AzureChatOpenAI
# model = AzureChatOpenAI(azure_deployment="gpt-4.1-mini", streaming=True)

# --- AWS Bedrock ---
# from langchain_aws import ChatBedrockConverse
# model = ChatBedrockConverse(
#     provider="anthropic",
#     model_id="anthropic.claude-sonnet-4-20250514-v1:0",
# )
