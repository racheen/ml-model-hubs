import streamlit as st

def render_page_header(title: str, description: str, category: str, task_type: str):
    st.title(title)
    st.caption(description)
    left, right = st.columns([3, 1])
    left.markdown(f"**Category folder:** `{category}`")
    right.markdown(f"**Task:** `{task_type.title()}`")

def render_section_title(title: str, help_text: str | None = None):
    st.subheader(title)
    if help_text:
        st.caption(help_text)
