import streamlit as st
from pages import home, clustering, real_estate , loan_eligibility , neural_networks

# Page configuration
st.set_page_config(
    page_title="ML Models Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

# Sidebar navigation
st.sidebar.title("ML Models Hub")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a Model:",
    [
        "Home",
        "Real Estate Pricing",
        "Loan Eligibility",
        "Customer Segmentation",
        "Neural Networks"
    ]
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

# Route to appropriate page
if page == "Home":
    home.show()
elif page == "Real Estate Pricing":
    real_estate.show()
elif page == "Loan Eligibility":
    loan_eligibility.show()
elif page == "Customer Segmentation":
    clustering.show()
elif page == "Neural Networks":
    neural_networks.show()