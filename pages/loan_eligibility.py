import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.model_manager import load_model, get_model_info

def show():
    """Display loan eligibility prediction page"""
    st.header("Loan Eligibility Prediction")
    
    st.markdown("""
    This model predicts whether a loan application will be approved based on:
    - **Applicant Income**
    - **Co-applicant Income**
    - **Loan Amount**
    - **Loan Term**
    - **Credit History**
    - **Employment Status**
    - **And more features...**
    """)
    
    # Load the loan eligibility models
    try:
        lr_model = load_model('Logistic_Regression', category='Loan Eligibility')
        rf_model = load_model('Random_Forest_Classifier', category='Loan Eligibility')
        
        if lr_model is None and rf_model is None:
            st.error("Loan Eligibility models not found. Please train them first.")
            st.stop()
        
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()
    
    # Display model info
    col1, col2 = st.columns(2)
    
    with col1:
        if lr_model:
            lr_info = get_model_info('Logistic_Regression', 'Loan Eligibility')
            if lr_info:
                st.metric("Logistic Regression Model", f"{lr_info['size_mb']:.2f} MB")
    
    with col2:
        if rf_model:
            rf_info = get_model_info('Random_Forest_Classifier', 'Loan Eligibility')
            if rf_info:
                st.metric("Random Forest Model", f"{rf_info['size_mb']:.2f} MB")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Single Application", "Batch Upload", "Model Comparison"])
    
    with tab1:
        st.markdown("### Predict Loan Eligibility")
        
        # Create input columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            applicant_income = st.number_input("Applicant Income ($)", min_value=0, max_value=100000, value=5000)
        with col2:
            coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, max_value=100000, value=0)
        with col3:
            loan_amount = st.number_input("Loan Amount ($1000s)", min_value=0, max_value=1000, value=128)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            loan_term = st.selectbox("Loan Term (months)", [60, 120, 180, 240, 300, 360, 480])
        with col5:
            credit_history = st.selectbox("Credit History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with col6:
            gender = st.selectbox("Gender", ["Male", "Female"])
        
        col7, col8, col9 = st.columns(3)
        
        with col7:
            married = st.selectbox("Married", ["Yes", "No"])
        with col8:
            dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        with col9:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        
        col10, col11, col12 = st.columns(3)
        
        with col10:
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        with col11:
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
        # Model selection
        st.markdown("---")
        model_choice = st.radio("Select Model for Prediction", 
                               ["Logistic Regression", "Random Forest", "Compare Both"],
                               horizontal=True)
        
        if st.button("Predict Eligibility", use_container_width=True):
            # Encode categorical variables
            gender_female = 1 if gender == "Female" else 0
            gender_male = 1 if gender == "Male" else 0
            married_yes = 1 if married == "Yes" else 0
            married_no = 1 if married == "No" else 0
            dependents_0 = 1 if dependents == "0" else 0
            dependents_1 = 1 if dependents == "1" else 0
            dependents_2 = 1 if dependents == "2" else 0
            dependents_3plus = 1 if dependents == "3+" else 0
            education_graduate = 1 if education == "Graduate" else 0
            education_not_graduate = 1 if education == "Not Graduate" else 0
            self_employed_yes = 1 if self_employed == "Yes" else 0
            self_employed_no = 1 if self_employed == "No" else 0
            property_rural = 1 if property_area == "Rural" else 0
            property_semiurban = 1 if property_area == "Semiurban" else 0
            property_urban = 1 if property_area == "Urban" else 0
            
            # Prepare input data with all required features
            input_data = np.array([[
                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_term,
                credit_history,
                gender_female,
                gender_male,
                married_no,
                married_yes,
                dependents_0,
                dependents_1,
                dependents_2,
                dependents_3plus,
                education_graduate,
                education_not_graduate,
                self_employed_no,
                self_employed_yes,
                property_rural,
                property_semiurban,
                property_urban
            ]])

            try:
                st.markdown("---")
                st.markdown("### Prediction Result")
                
                lr_prediction = None
                rf_prediction = None
                lr_probability = None
                rf_probability = None
                
                if model_choice in ["Logistic Regression", "Compare Both"] and lr_model:
                    lr_prediction = lr_model.predict(input_data)[0]
                    lr_probability = lr_model.predict_proba(input_data)[0][1]
                    
                    status = "APPROVED" if lr_prediction == 1 else "REJECTED"
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Logistic Regression", status)
                    with col2:
                        st.metric("Approval Probability", f"{lr_probability*100:.2f}%")
                
                if model_choice in ["Random Forest", "Compare Both"] and rf_model:
                    rf_prediction = rf_model.predict(input_data)[0]
                    rf_probability = rf_model.predict_proba(input_data)[0][1]
                    
                    status = "APPROVED" if rf_prediction == 1 else "REJECTED"
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Random Forest", status)
                    with col2:
                        st.metric("Approval Probability", f"{rf_probability*100:.2f}%")
                
                if model_choice == "Compare Both" and lr_prediction is not None and rf_prediction is not None:
                    st.markdown("---")
                    if lr_prediction == rf_prediction:
                        st.info("Both models agree on the prediction!")
                    else:
                        st.warning("Models disagree on the prediction. Review carefully.")
                    
                    avg_probability = (lr_probability + rf_probability) / 2
                    st.info(f"**Average Approval Probability:** {avg_probability*100:.2f}%")
                
                # Display application details
                st.markdown("### Application Details")
                details = f"""
                **Income Information:**
                - **Applicant Income:** ${applicant_income:,}
                - **Co-applicant Income:** ${coapplicant_income:,}
                - **Total Income:** ${applicant_income + coapplicant_income:,}
                
                **Loan Information:**
                - **Loan Amount:** ${loan_amount * 1000:,}
                - **Loan Term:** {loan_term} months
                - **Credit History:** {"Yes" if credit_history == 1 else "No"}
                
                **Personal Information:**
                - **Gender:** {gender}
                - **Married:** {married}
                - **Dependents:** {dependents}
                - **Education:** {education}
                - **Self Employed:** {self_employed}
                - **Property Area:** {property_area}
                """
                st.markdown(details)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with tab2:
        st.markdown("### Predict Multiple Applications")
        
        uploaded_file = st.file_uploader("Upload CSV file with loan applications", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                
                # Check required columns
                required_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
                               'Credit_History', 'Gender_Female', 'Gender_Male', 'Married_No', 'Married_Yes',
                               'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+',
                               'Education_Graduate', 'Education_Not Graduate', 'Self_Employed_No', 'Self_Employed_Yes',
                               'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']
                
                missing_cols = [col for col in required_cols if col not in df_upload.columns]
                
                if missing_cols:
                    st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                    st.stop()
                
                # Select model for batch prediction
                batch_model = st.radio("Select Model", 
                                      ["Logistic Regression", "Random Forest"],
                                      horizontal=True)
                
                if st.button("Predict Eligibility", use_container_width=True):
                    X = df_upload[required_cols].values
                    
                    if batch_model == "Logistic Regression" and lr_model:
                        predictions = lr_model.predict(X)
                        probabilities = lr_model.predict_proba(X)[:, 1]
                        model_name = "Logistic Regression"
                    else:
                        predictions = rf_model.predict(X)
                        probabilities = rf_model.predict_proba(X)[:, 1]
                        model_name = "Random Forest"
                    
                    df_upload['Prediction'] = predictions.map({1: 'APPROVED', 0: 'REJECTED'})
                    df_upload['Approval_Probability'] = probabilities
                    
                    st.success(f"Predicted eligibility for {len(df_upload)} applications using {model_name}!")
                    
                    # Display results
                    st.dataframe(df_upload, use_container_width=True)
                    
                    # Download results
                    csv = df_upload.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="loan_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Display statistics
                    st.markdown("---")
                    st.markdown("### Prediction Statistics")
                    
                    approved_count = (df_upload['Prediction'] == 'APPROVED').sum()
                    rejected_count = (df_upload['Prediction'] == 'REJECTED').sum()
                    approval_rate = (approved_count / len(df_upload)) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Applications", len(df_upload))
                    with col2:
                        st.metric("Approved", approved_count)
                    with col3:
                        st.metric("Rejected", rejected_count)
                    
                    st.metric("Approval Rate", f"{approval_rate:.2f}%")
                    
                    # Visualization
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Pie chart
                    axes[0].pie([approved_count, rejected_count], 
                               labels=['Approved', 'Rejected'],
                               autopct='%1.1f%%',
                               colors=['#2ecc71', '#e74c3c'])
                    axes[0].set_title('Loan Approval Distribution')
                    
                    # Probability distribution
                    axes[1].hist(df_upload['Approval_Probability'], bins=20, color='#3498db', edgecolor='black')
                    axes[1].set_xlabel('Approval Probability')
                    axes[1].set_ylabel('Frequency')
                    axes[1].set_title('Distribution of Approval Probabilities')
                    
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.markdown("### Model Comparison")
        
        if lr_model and rf_model:
            st.markdown("Compare the performance and characteristics of both models.")
            
            # Model information
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Logistic Regression")
                lr_info = get_model_info('Logistic_Regression', 'Loan Eligibility')
                if lr_info:
                    st.info(f"""
                    **Model Size:** {lr_info['size_mb']:.2f} MB
                    **Type:** Linear Classification
                    **Training Date:** {lr_info.get('date', 'N/A')}
                    """)
            
            with col2:
                st.markdown("#### Random Forest")
                rf_info = get_model_info('Random_Forest_Classifier', 'Loan Eligibility')
                if rf_info:
                    st.info(f"""
                    **Model Size:** {rf_info['size_mb']:.2f} MB
                    **Type:** Ensemble Classification
                    **Training Date:** {rf_info.get('date', 'N/A')}
                    """)
            
            st.markdown("---")
            st.markdown("#### Model Characteristics")
            
            comparison_data = {
                'Characteristic': [
                    'Algorithm Type',
                    'Interpretability',
                    'Training Speed',
                    'Prediction Speed',
                    'Handles Non-linearity',
                    'Feature Importance',
                    'Overfitting Risk'
                ],
                'Logistic Regression': [
                    'Linear',
                    'High',
                    'Fast',
                    'Very Fast',
                    'No',
                    'Coefficients',
                    'Low'
                ],
                'Random Forest': [
                    'Ensemble',
                    'Medium',
                    'Moderate',
                    'Moderate',
                    'Yes',
                    'Built-in',
                    'Medium'
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("#### When to Use Each Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Use Logistic Regression when:**
                - You need high interpretability
                - You have limited computational resources
                - Your data is linearly separable
                - You need fast predictions
                - You want to understand feature coefficients
                """)
            
            with col2:
                st.markdown("""
                **Use Random Forest when:**
                - You have complex non-linear relationships
                - You need high accuracy
                - You have sufficient computational resources
                - You want automatic feature importance
                - You can tolerate slightly longer prediction times
                """)
        else:
            st.warning("Both models need to be loaded for comparison.")

if __name__ == "__main__":
    show()
