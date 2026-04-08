# Streamlit Frontend

import base64
import os
from io import BytesIO
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from httpx import HTTPStatusError

from utils import handle_file_upload, get_messages, get_streaming_response

load_dotenv()

# Layout
st.set_page_config(
    layout="wide",
    initial_sidebar_state="auto"
)

FASTAPI_URL = os.getenv("FASTAPI_URL")

with st.sidebar:
    st.image("https://api.dicebear.com/8.x/adventurer/svg?seed=dummy", width=50)
    st.write("Logged in as: **Dummy User**")
    st.write("Role: Administrator")
    st.divider()
    if st.button("Log out"):
        st.warning("Logged out")

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

"""
Session states - session_id, messages, file upload and download links
"""
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

prompt = st.chat_input(
    placeholder="Hi! How can I help you?",
    accept_file="multiple",
    file_type=["pdf", "txt", "csv", "xls", "xlsx"]
)

with chat_container:
    if prompt:
        # Extract text if file uploads are enabled
        user_message = prompt.text
        
        attached = [{"name": f.name, "type": f.type, "bytes": f.read()} for f in (prompt["files"] or [])]

        session_id = str(st.session_state.session_id)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_message,
                "files": [{"name": f["name"]} for f in attached]
            }
        )

        with st.chat_message("user"):
            st.markdown(user_message)
            if attached:
                st.caption(f"📎 {len(attached)} file(s) attached")

        if attached:
            files_payload = [("files", (f["name"], f["bytes"], f["type"])) for f in attached]

            try:
                response = handle_file_upload(
                    url=f"{FASTAPI_URL}/uploads",
                    data={"session_id": session_id},
                    files_payload=files_payload
                )
                response.raise_for_status()
            except HTTPStatusError as e:
                st.error(
                    body=f"File upload failed: {e.response.status_code} - {e.response.text}",
                    icon="❌"
                )
                st.stop()

        # Define payloads to sent to end-points
        payload_chat = {
            "session_id": session_id,
            "message": user_message, # adds current user prompt
        }

        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    charts_buffer = []
                    full_response = st.write_stream(
                        get_streaming_response(
                            url=f"{FASTAPI_URL}/chat",
                            payload=payload_chat,
                            charts_out=charts_buffer
                        )
                    )
            print(f"charts_buffer after write_stream: {len(charts_buffer)}")
            for i, chart in enumerate(charts_buffer):
                # Plots and charts arrive as byte stream
                # Must be decoded to reconstruct image
                img_bytes = base64.b64decode(chart["data"])
                st.image(BytesIO(img_bytes), width=500)
                st.download_button(
                    label="Download chart",
                    data=img_bytes,
                    file_name=f"chart_{i+1}.png",
                    key=f"download_chart_{session_id}_{i}",
                    icon=":material/download:"
                )

            assistant_msg = {
                "role": "assistant",
                "content": full_response
            }
            st.session_state.messages.append(assistant_msg)
        except HTTPStatusError as e:
            st.error(
                body=f"Streaming failed: {e.response.status_code} - {e.response.text}",
                icon="❌"
            )
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error during streaming: {e}")
            st.stop()