import streamlit as st
from app.components.common.display import show_dataframe, show_metric_cards
from app.components.common.layout import render_section_title
from app.core.state_manager import get_page_value
from app.services.comparison_service import compare_metric_sets

def render_compare_section():
    render_section_title("Compare Default vs Custom", "Use this after training a custom cluster model.")
    default_prediction = get_page_value("clustering", "default_prediction")
    custom_result = get_page_value("clustering", "custom_training_result")
    if custom_result is None:
        st.info("Train a custom clustering model first.")
        return
    st.markdown("**Custom model metrics**")
    show_metric_cards(custom_result.metrics)
    comparison = compare_metric_sets({"saved_model": "manual evaluation needed"}, custom_result.metrics)
    st.markdown("**Metric comparison**")
    show_dataframe(comparison.summary_table)
    if default_prediction is not None:
        st.write(f"Default cluster label from prediction section: **{default_prediction}**")
