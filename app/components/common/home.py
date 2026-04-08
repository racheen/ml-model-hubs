import streamlit as st
from pathlib import Path
from app.config.settings import APP_TITLE
from app.config.paths import MODELS_DIR


def _count_models() -> int:
    if not MODELS_DIR.exists():
        return 0
    return len(list(Path(MODELS_DIR).rglob("*_model.pkl")))


def render_home():
  

    total_models = _count_models()
    total_modules = 4
    task_types = 4
    total_default_models = 6

    st.title(APP_TITLE)
    st.caption("A modular machine learning hub for prediction, analysis, tuning, and comparison.")

    st.divider()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Modules", total_modules)
    kpi2.metric("Default Models", total_default_models)
    kpi3.metric("Total Models", total_models)
    kpi4.metric("Workflows", task_types)

    st.divider()

    st.subheader("Platform Overview")
    st.write(
        """
        This application brings together four machine learning workflows in one place.
        Each module is organized around a consistent structure so users can explore
        default models, review analysis, tune model settings, and compare results.
        """
    )

    overview_col1, overview_col2 = st.columns([1.2, 1])

    with overview_col1:
        st.markdown("#### Available Modules")
        module_data = [
            {
                "module": "Loan Eligibility",
                "task": "Classification",
                "focus": "Predict loan approval outcomes from applicant information.",
            },
            {
                "module": "Real Estate",
                "task": "Regression",
                "focus": "Estimate property values and examine feature relationships.",
            },
            {
                "module": "Student Admission",
                "task": "Neural Network Prediction",
                "focus": "Model admission likelihood using academic profile inputs.",
            },
            {
                "module": "Customer Segmentation",
                "task": "Clustering",
                "focus": "Group customers into segments based on shared characteristics.",
            },
        ]
        st.dataframe(module_data, use_container_width=True, hide_index=True)

    with overview_col2:
        st.markdown("#### How the Hub Is Organized")
        st.info(
            """
            Use the sidebar to open any module page.

            Each module is designed around:
            - prediction with default models
            - model analysis
            - parameter tuning
            - result comparison
            """
        )

    st.divider()

    st.subheader("Module Summaries")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("#### Loan Eligibility")
            st.write(
                "Classification workflow focused on predicting whether an applicant is likely to be approved for a loan."
            )

        with st.container(border=True):
            st.markdown("#### Real Estate")
            st.write(
                "Regression workflow for analyzing housing data and predicting estimated property prices."
            )

    with col2:
        with st.container(border=True):
            st.markdown("#### Student Admission")
            st.write(
                "Neural network workflow for estimating admission outcomes from academic and profile-based inputs."
            )

        with st.container(border=True):
            st.markdown("#### Customer Segmentation")
            st.write(
                "Clustering workflow for discovering customer groups and understanding segment behavior."
            )

    st.divider()

    st.subheader("About This App")
    st.write(
        """
        The hub is built to present machine learning workflows in a readable and modular format.
        It is intended for exploring trained default models while also allowing controlled tuning
        and comparison inside each module page.
        """
    )