import json
from typing import Generator

import httpx
import streamlit as st

@st.cache_resource
def get_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages

def handle_file_upload(url: str, data: dict, files_payload: list) -> httpx.Response:
    """Function for handling file uploads"""
    with httpx.Client() as client:
        response = client.post(
            url,
            data=data,
            files=files_payload
        )

    return response

def get_streaming_response(url: str, payload: dict[str, str], charts_out: list) -> Generator[str, None, None]:
    """Function for streaming LLM response."""
    with httpx.stream("POST", url=url, json=payload, timeout=120) as response:
        response.raise_for_status()

        for chunk in response.iter_lines():
            if chunk:
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "token":
                    yield data["content"] if "content" in data else None
                elif data.get("type") == "charts":
                    print(f"Received {len(data['charts'])} chart(s) in stream")
                    charts_out.extend(data["charts"])
