import streamlit as st
from app.components.common.display import show_dataframe, show_metric_cards
from app.components.common.layout import render_section_title
from app.core.state_manager import get_page_value
from app.services.comparison_service import compare_metric_sets, compare_predictions

def render_compare_section():
    render_section_title("Compare Default vs Custom", "Use this after training a custom model.")
    default_prediction = get_page_value("real_estate", "default_prediction")
    custom_result = get_page_value("real_estate", "custom_training_result")
    if custom_result is None:
        st.info("Train a custom model first.")
        return
    st.markdown("**Custom model metrics**")
    show_metric_cards(custom_result.metrics)
    comparison = compare_metric_sets({"saved_model": "manual evaluation needed"}, custom_result.metrics)
    st.markdown("**Metric comparison**")
    show_dataframe(comparison.summary_table)
    if default_prediction is not None:
        prediction_table = compare_predictions(default_prediction, "Run custom inference on your uploaded data")
        st.markdown("**Prediction comparison**")
        show_dataframe(prediction_table)
    else:
        st.info("Run a default prediction first to compare outputs.")
