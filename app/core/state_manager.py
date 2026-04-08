import streamlit as st

def get_page_state(page_key: str) -> dict:
    if page_key not in st.session_state:
        st.session_state[page_key] = {}
    return st.session_state[page_key]

def set_page_value(page_key: str, key: str, value):
    page_state = get_page_state(page_key)
    page_state[key] = value
    st.session_state[page_key] = page_state

def get_page_value(page_key: str, key: str, default=None):
    page_state = get_page_state(page_key)
    return page_state.get(key, default)
