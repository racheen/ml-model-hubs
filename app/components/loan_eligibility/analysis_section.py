import streamlit as st
from app.components.common.analysis_helpers import render_default_model_analysis
from app.components.common.layout import render_section_title
from app.domain.loan_eligibility.config import CATEGORY, DEFAULT_MODELS, USE_JOBLIB, EXPECTED_FEATURES

def render_analysis_section():
    render_section_title("Model Analysis", "Inspect saved model files and any exposed coefficients or feature importances.")
    model_name = st.selectbox("Model to analyze", DEFAULT_MODELS, key="loan_eligibility_analysis_model")
    render_default_model_analysis(
        category=CATEGORY,
        model_name=model_name,
        feature_names=EXPECTED_FEATURES,
        use_joblib=USE_JOBLIB,
    )
