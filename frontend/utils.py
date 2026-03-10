import json
from typing import Generator

import httpx
import streamlit as st

@st.cache_resource
def get_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages

# def get_streaming_response(url: str, payload: dict[str, str]) -> Generator[str]:
#     """Function for streaming LLM response."""
#     with httpx.stream("POST", url=url, payload=payload, timeout=120) as response:
#         response.raise_for_status()
#
#         for chunk in response.iter_text():
#             if chunk:
#                 data = json.loads(chunk)
#                 if data.get("type") == "token":
#                     yield data["content"] if "content" in data else None