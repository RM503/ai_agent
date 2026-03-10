# Streamlit Frontend
import json
import os
from typing import Generator
from uuid import uuid4

import httpx
import streamlit as st
from dotenv import load_dotenv
from httpx import HTTPStatusError

from utils import get_messages

load_dotenv()

FASTAPI_URL = os.getenv("FASTAPI_URL")

st.title("Welcome user")

st.write(
    """
    Q is an an AI agent currently in its development stage. Other than serving
    as a general chatbot, it is equipped to perform the following tasks:
    
    - Transcription of audio files with summarization
    - Research using custom data (RAG agent)
    - Data analysis
    """
)

# Store session_id and messages
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

# Chat UI
st.session_state.messages = get_messages()

chat_container = st.container(height=400)
with chat_container:
    for message in st.session_state.messages:
        content = message.get("content")

        if not isinstance(content, str):
            try:
                content = content.text
            except AttributeError:
                continue

        with st.chat_message(message["role"]):
            st.markdown(content)

prompt = st.chat_input("Hi! How can I help you?", accept_file=True)

with chat_container:
    if prompt:
        # Extract text if file uploads are enabled
        user_message = prompt.text
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )
        with st.chat_message("user"):
            st.markdown(user_message)

        payload = {
            "session_id": str(st.session_state.session_id),
            "message": user_message, # adds current user prompt
        }


        def get_streaming_response() -> Generator[str]:
            """Function for streaming LLM response."""
            with httpx.stream("POST", url=f"{FASTAPI_URL}/chat", json=payload, timeout=120) as response:
                response.raise_for_status()

                for chunk in response.iter_text():
                    if chunk:
                        data = json.loads(chunk)
                        if data.get("type") == "token":
                            yield data["content"] if "content" in data else None

        try:
            with st.chat_message("assistant"):
                full_response = st.write_stream(get_streaming_response())

            assistant_msg = {
                "role": "assistant",
                "content": full_response
            }
            st.session_state.messages.append(assistant_msg)
        except HTTPStatusError as e:
            raise e