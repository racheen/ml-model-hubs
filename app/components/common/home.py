import streamlit as st
from app.config.settings import APP_TITLE
from app.config.paths import MODELS_DIR

def render_home():
    st.set_page_config(
        page_title="ML Models Hub",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("ML Models Hub")
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(
    """
    A modular Streamlit app for exploring four machine-learning workflows:

    - Loan Eligibility
    - Real Estate
    - Student Admission
    - Customer Segmentation

    Use the left sidebar to open a page.
    """
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About This App
    Multi-model machine learning application featuring:
    - Real Estate Price Prediction
    - Loan Eligibility Classification
    - Customer Segmentation
    - Neural Network Models

    **Built with:** Streamlit | Scikit-learn | TensorFlow
    """)
