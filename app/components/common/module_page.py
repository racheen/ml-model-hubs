import streamlit as st
from app.components.common.layout import render_page_header

def render_module_page(config, render_predict, render_analysis, render_train=None, render_compare=None):
    render_page_header(
        title=config["TITLE"],
        description=config["DESCRIPTION"],
        category=config["CATEGORY"],
        task_type=config["TASK_TYPE"],
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Predict",
        "Model Analysis",
        "Tune / Custom Model",
        "Compare"
    ])

    with tab1:
        render_predict()

    with tab2:
        render_analysis()

    with tab3:
        render_train()

    with tab4:
        render_compare()