import streamlit as st

@st.cache_resource
def get_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages