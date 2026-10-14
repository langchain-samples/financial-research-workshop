"""Centralized model initialization.

The notebooks all import `model` from here, so swapping providers
(OpenAI / Anthropic / Azure / Bedrock) only requires editing this file.

The default is OpenAI, direct. Module 2 walks through introducing the
LangSmith LLM Gateway as the production-ready alternative — to make the
switch, comment out the default below and uncomment the gateway block.
"""

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env", override=True)

from langchain.chat_models import init_chat_model

# --- Default: OpenAI, direct ---
# model = init_chat_model("openai:gpt-4.1-mini")

# --- OpenAI via the LangSmith LLM Gateway (Module 2 §1.4) ---
# Routes every model call through the LangSmith Gateway so that workspace
# policies (PII / secrets / allow-lists / cost caps) are enforced.
model = init_chat_model(
    model="gpt-4.1-mini",
    model_provider="openai",
    base_url="https://gateway.smith.langchain.com/openai",
    api_key=os.environ["LANGSMITH_API_KEY_GATEWAY"],
)

# --- Anthropic ---
# model = init_chat_model("anthropic:claude-sonnet-4-6")

# --- Azure OpenAI ---
# from langchain_openai import AzureChatOpenAI
# model = AzureChatOpenAI(azure_deployment="gpt-4.1-mini", streaming=True)

# --- AWS Bedrock ---
# from langchain_aws import ChatBedrockConverse
# model = ChatBedrockConverse(
#     provider="openai",
#     model_id="openai.gpt-4.1-mini",
# )
