from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)