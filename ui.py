import streamlit as st
import time
from constant import TEMPLATES

def show_thinking_animation():
    with st.empty():
        for i in range(2):
            dots = "." * (i % 4)
            st.markdown(f"🤔 {TEMPLATES['initial_think']}{dots}")
            time.sleep(0.5)

def render_chat_history(history):
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
