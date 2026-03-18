import streamlit as st
import pandas as pd

def show():
    """Display home page with overview of all models"""
    
    st.title("Machine Learning Models Hub")
    
    st.markdown("""
    Welcome to the comprehensive ML Models Hub! This application showcases multiple 
    machine learning models trained on different datasets and use cases.
    """)
    
    st.markdown("---")
    
    # Models overview
    st.header("Available Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Real Estate Price Prediction
        **Task:** Regression
        
        Predict property prices using features like:
        - Square footage
        - Number of bedrooms/bathrooms
        - Year built
        - Property tax & insurance
        
        **Models:** Linear Regression, Random Forest
        """)
        
        st.markdown("""
        ### Loan Eligibility Classification
        **Task:** Binary Classification
        
        Determine loan approval likelihood based on:
        - Applicant income
        - Credit history
        - Employment status
        - Marital status
        
        **Models:** Logistic Regression, Random Forest
        """)
    
    with col2:
        st.markdown("""
        ### Customer Segmentation
        **Task:** Unsupervised Clustering
        
        Segment mall customers into groups based on:
        - Age
        - Annual income
        - Spending score
        
        **Models:** K-Means Clustering
        """)
        
        st.markdown("""
        ### Neural Network Models
        **Task:** Deep Learning
        
        Advanced neural network implementations for:
        - Complex pattern recognition
        - Multi-layer architectures
        - Non-linear relationships
        
        **Models:** TensorFlow/Keras Networks
        """)
    
    st.markdown("---")
    
    # Quick stats
    st.header("Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", "6")
    with col2:
        st.metric("Model Types", "3")
    with col3:
        st.metric("Datasets", "4")
    with col4:
        st.metric("Features", "50+")
    
    st.markdown("---")
    
    # How to use
    st.header("How to Use")
    
    st.markdown("""
    1. **Select a Model** from the sidebar
    2. **View Model Info** - See model details and performance metrics
    3. **Make Predictions** - Input data and get predictions
    4. **Analyze Results** - View visualizations and insights
    
    Each model page includes:
    - Model information and statistics
    - Single prediction interface
    - Batch prediction capability
    - Performance metrics and visualizations
    """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: gray; margin-top: 3rem;'>
    <p>Built using Streamlit | ML2 Project - Algonquin College</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show()
