import streamlit as st

from app.components.common.home import render_home
from app.components.loan_eligibility.page import render_page as render_loan_page
from app.components.real_estate.page import render_page as render_real_estate_page
from app.components.student_admission.page import render_page as render_neural_networks_page
from app.components.clustering.page import render_page as render_clustering_page

st.set_page_config(
    page_title="ML Models Hub",
    layout="wide",
    initial_sidebar_state="expanded",
)

selected_page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Loan Eligibility",
        "Real Estate",
        "Student Admission",
        "Customer Segmentation",
    ],
)

if selected_page == "Home":
    render_home()
elif selected_page == "Loan Eligibility":
    render_loan_page()
elif selected_page == "Real Estate":
    render_real_estate_page()
elif selected_page == "Student Admission":
    render_neural_networks_page()
elif selected_page == "Customer Segmentation":
    render_clustering_page()
