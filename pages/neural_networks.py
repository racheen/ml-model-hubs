
import streamlit as st
import numpy as np
import pandas as pd
from utils.model_manager import load_model, get_model_info, load_scaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def show():
    """Display Neural Networks page for UCLA admission prediction"""
    st.header("UCLA Admission Prediction (Neural Networks)")
    
    st.markdown("""
    This model predicts a student's chances of admission to UCLA using a Neural Network.
    
    **Prediction based on:**
    - GRE Score (out of 340)
    - TOEFL Score (out of 120)
    - University Rating (1-5)
    - Statement of Purpose (1-5)
    - Letter of Recommendation (1-5)
    - CGPA (out of 10)
    - Research Experience (Yes/No)
    """)
    
    # Load the neural network model
    try:
        model = load_model('Neural_Network', category='Neural Networks', use_joblib=True)
        if model is None:
            st.error("Neural Network model not found. Please train it first.")
            st.stop()
        st.success("Neural Network model loaded!")
    except TypeError:
        # Fallback if category parameter not supported
        model = load_model('Neural_Network', use_joblib=True)
        if model is None:
            st.error("Neural Network model not found. Please train it first.")
            st.stop()
        st.success("Neural Network model loaded!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Display model info
    model_info = get_model_info('Neural_Network', 'Neural Networks', use_joblib=True)
    if model_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", "MLPClassifier")
        with col2:
            st.metric("Model Size", f"{model_info['size_mb']:.2f} MB")
        with col3:
            st.metric("Hidden Layers", "1 (3 neurons)")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Single Student", "Batch Upload", "Model Analysis"])
    
    with tab1:
        st.markdown("### Predict Single Student Admission")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gre_score = st.number_input("GRE Score (0-340)", min_value=0, max_value=340, value=320)
            toefl_score = st.number_input("TOEFL Score (0-120)", min_value=0, max_value=120, value=110)
            university_rating = st.slider("University Rating (1-5)", min_value=1, max_value=5, value=3)
            sop = st.slider("Statement of Purpose (1-5)", min_value=1, max_value=5, value=3)
        
        with col2:
            lor = st.slider("Letter of Recommendation (1-5)", min_value=1, max_value=5, value=3)
            cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, value=8.5, step=0.1)
            research = st.selectbox("Research Experience", ["No", "Yes"])
        
        if st.button("Predict Admission Chance", use_container_width=True):
            try:
                # Prepare input data
                research_val = 1 if research == "Yes" else 0
                
                # Create feature array with one-hot encoding for University_Rating and Research
                # Features order: GRE_Score, TOEFL_Score, SOP, LOR, CGPA, 
                # University_Rating_1, University_Rating_2, University_Rating_3, University_Rating_4, University_Rating_5,
                # Research_0, Research_1
                
                input_features = [
                    gre_score,
                    toefl_score,
                    sop,
                    lor,
                    cgpa
                ]
                
                # One-hot encode University Rating
                for i in range(1, 6):
                    input_features.append(1 if university_rating == i else 0)
                
                # One-hot encode Research
                input_features.append(1 if research_val == 0 else 0)  # Research_0
                input_features.append(1 if research_val == 1 else 0)  # Research_1
                
                input_data = np.array([input_features])
                
                # Scale the input (using standard scaling based on training data)
                scaler = load_scaler(category='Neural Networks')
                input_scaled = scaler.fit_transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                
                st.markdown("---")
                st.markdown("### Admission Prediction Result")
                
                # Display result with color coding
                if prediction == 1:
                    st.success(f"**LIKELY TO BE ADMITTED**")
                    admission_prob = probability[1] * 100
                else:
                    st.error(f"**UNLIKELY TO BE ADMITTED**")
                    admission_prob = probability[0] * 100
                
                # Display probabilities
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Admission Probability", f"{probability[1]*100:.2f}%")
                with col2:
                    st.metric("Rejection Probability", f"{probability[0]*100:.2f}%")
                
                # Display student profile
                st.markdown("### Student Profile Summary")
                profile = f"""
                **Academic Scores:**
                - GRE Score: {gre_score}/340
                - TOEFL Score: {toefl_score}/120
                - CGPA: {cgpa}/10.0
                
                **Application Strength:**
                - University Rating: {university_rating}/5
                - Statement of Purpose: {sop}/5
                - Letter of Recommendation: {lor}/5
                - Research Experience: {research}
                """
                st.markdown(profile)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(8, 5))
                categories = ['Admitted', 'Not Admitted']
                probs = [probability[1]*100, probability[0]*100]
                colors = ['#2ecc71', '#e74c3c']
                bars = ax.bar(categories, probs, color=colors, alpha=0.7, edgecolor='black')
                ax.set_ylabel('Probability (%)')
                ax.set_title('Admission Prediction Probability')
                ax.set_ylim(0, 100)
                
                # Add value labels on bars
                for bar, prob in zip(bars, probs):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{prob:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with tab2:
        st.markdown("### Predict Multiple Students")
        
        uploaded_file = st.file_uploader("Upload CSV file with student data", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                
                # Check required columns
                required_cols = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA', 'Research']
                missing_cols = [col for col in required_cols if col not in df_upload.columns]
                
                if missing_cols:
                    st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                    st.stop()
                
                if st.button("Predict Admissions", use_container_width=True):
                    # Prepare data
                    predictions_list = []
                    probabilities_list = []
                    
                    scaler = MinMaxScaler()
                    
                    for idx, row in df_upload.iterrows():
                        # Create feature array
                        input_features = [
                            row['GRE_Score'],
                            row['TOEFL_Score'],
                            row['SOP'],
                            row['LOR'],
                            row['CGPA']
                        ]
                        
                        # One-hot encode University Rating
                        for i in range(1, 6):
                            input_features.append(1 if row['University_Rating'] == i else 0)
                        
                        # One-hot encode Research
                        research_val = 1 if row['Research'] == 1 else 0
                        input_features.append(1 if research_val == 0 else 0)
                        input_features.append(1 if research_val == 1 else 0)
                        
                        input_data = np.array([input_features])
                        input_scaled = scaler.fit_transform(input_data)
                        
                        pred = model.predict(input_scaled)[0]
                        prob = model.predict_proba(input_scaled)[0]
                        
                        predictions_list.append('Admitted' if pred == 1 else 'Not Admitted')
                        probabilities_list.append(prob[1])
                    
                    df_upload['Prediction'] = predictions_list
                    df_upload['Admission_Probability'] = probabilities_list
                    
                    st.success(f"✓ Predicted admission for {len(df_upload)} students!")
                    
                    # Display results
                    st.dataframe(df_upload, use_container_width=True)
                    
                    # Download results
                    csv = df_upload.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="admission_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Statistics
                    st.markdown("---")
                    st.markdown("### Prediction Statistics")
                    
                    admitted_count = (df_upload['Prediction'] == 'Admitted').sum()
                    not_admitted_count = (df_upload['Prediction'] == 'Not Admitted').sum()
                    admission_rate = (admitted_count / len(df_upload)) * 100
                    avg_probability = df_upload['Admission_Probability'].mean() * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Students", len(df_upload))
                    with col2:
                        st.metric("Admitted", admitted_count)
                    with col3:
                        st.metric("Not Admitted", not_admitted_count)
                    with col4:
                        st.metric("Admission Rate", f"{admission_rate:.1f}%")
                    
                    st.metric("Avg Admission Probability", f"{avg_probability:.2f}%")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.pie([admitted_count, not_admitted_count],
                              labels=['Admitted', 'Not Admitted'],
                              autopct='%1.1f%%',
                              colors=['#2ecc71', '#e74c3c'],
                              startangle=90)
                        ax.set_title('Admission Distribution')

                        ax.set_title('Admission Distribution')
                        st.pyplot(fig)
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.hist(df_upload['Admission_Probability'] * 100, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
                        ax.set_xlabel('Admission Probability (%)')
                        ax.set_ylabel('Number of Students')
                        ax.set_title('Distribution of Admission Probabilities')
                        ax.axvline(avg_probability, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_probability:.1f}%')
                        ax.legend()
                        st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.markdown("### Model Analysis & Performance")
        
        # Model Information
        st.markdown("#### Model Architecture")
        model_architecture = """
        **Neural Network Configuration:**
        - **Algorithm:** Multi-layer Perceptron (MLP) Classifier
        - **Input Features:** 12 (after one-hot encoding)
        - **Hidden Layer:** 1 layer with 3 neurons
        - **Activation Function:** tanh
        - **Output Layer:** 1 neuron (binary classification)
        - **Solver:** adam
        - **Max Iterations:** 200
        - **Batch Size:** 50
        """
        st.markdown(model_architecture)
        
        # Feature Importance
        st.markdown("#### Input Features")
        features_info = pd.DataFrame({
            'Feature': [
                'GRE Score',
                'TOEFL Score',
                'Statement of Purpose',
                'Letter of Recommendation',
                'CGPA',
                'University Rating (1-5)',
                'Research Experience'
            ],
            'Range/Type': [
                '0-340',
                '0-120',
                '1-5',
                '1-5',
                '0-10',
                'Categorical (1-5)',
                'Binary (Yes/No)'
            ],
            'Importance': [
                'High',
                'High',
                'Medium',
                'Medium',
                'High',
                'Medium',
                'Low-Medium'
            ]
        })
        st.dataframe(features_info, use_container_width=True)
        
        # Model Performance Metrics
        st.markdown("#### Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Type", "Classification")
            st.metric("Output", "Binary (Admitted/Not Admitted)")
            st.metric("Prediction Method", "Probability-based")
        
        with col2:
            st.metric("Training Algorithm", "Adam Optimizer")
            st.metric("Regularization", "L2 (default)")
            st.metric("Early Stopping", "Not enabled")
        
        # Key Insights
        st.markdown("#### Key Insights")
        insights = """
        **Model Strengths:**
        - ✓ Captures non-linear relationships between features
        - ✓ Provides probability estimates for each prediction
        - ✓ Relatively fast training and inference
        - ✓ Good generalization with proper regularization
        
        **Factors Influencing Admission:**
        1. **GRE Score** - Strong predictor of academic capability
        2. **TOEFL Score** - Important for international students
        3. **CGPA** - Demonstrates consistent academic performance
        4. **University Rating** - Indicates institution prestige
        5. **Research Experience** - Shows initiative and specialization
        6. **SOP & LOR** - Qualitative measures of student potential
        
        **Model Limitations:**
        - ⚠ Trained on historical data; may not capture recent admission trends
        - ⚠ Does not account for specific program requirements
        - ⚠ Limited to features provided in training data
        - ⚠ Predictions are probabilistic, not deterministic
        """
        st.markdown(insights)
        
        # Prediction Guidelines
        st.markdown("#### Interpretation Guidelines")
        
        interpretation = """
        **Admission Probability Ranges:**
        
        | Probability Range | Interpretation | Recommendation |
        |---|---|---|
        | 80-100% | Very High Chance | Strong candidate, likely to be admitted |
        | 60-80% | High Chance | Good candidate, competitive |
        | 40-60% | Moderate Chance | Borderline candidate, depends on other factors |
        | 20-40% | Low Chance | Weak candidate, consider improvements |
        | 0-20% | Very Low Chance | Unlikely to be admitted with current profile |
        
        **How to Improve Admission Chances:**
        - Increase GRE Score (target: 320+)
        - Improve TOEFL Score (target: 110+)
        - Maintain high CGPA (target: 8.5+)
        - Gain research experience
        - Write strong Statement of Purpose
        - Obtain strong Letters of Recommendation
        """
        st.markdown(interpretation)
        
        # Comparison with other models
        st.markdown("#### Model Comparison")
        
        comparison_data = {
            'Model': ['Neural Network', 'Logistic Regression', 'Random Forest', 'SVM'],
            'Complexity': ['High', 'Low', 'Medium', 'Medium'],
            'Interpretability': ['Low', 'High', 'Medium', 'Low'],
            'Speed': ['Fast', 'Very Fast', 'Medium', 'Medium'],
            'Non-linear': ['Yes', 'No', 'Yes', 'Yes']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        st.info("""
        **Why Neural Networks?**
        Neural Networks are chosen for this task because they:
        - Can capture complex non-linear relationships between admission factors
        - Provide probability estimates useful for decision-making
        - Scale well with additional features
        - Perform well on mixed data types (continuous and categorical)
        """)
        
        # Data Statistics
        st.markdown("#### Training Data Statistics")
        
        data_stats = """
        **Dataset Overview:**
        - Total Samples: 500 students
        - Training Set: 400 students (80%)
        - Test Set: 100 students (20%)
        - Features: 7 input features
        - Target: Binary (Admitted/Not Admitted)
        
        **Class Distribution:**
        - Admitted: ~60%
        - Not Admitted: ~40%
        """
        st.markdown(data_stats)
        
        # Usage Tips
        st.markdown("#### Usage Tips")
        
        tips = """
        **Best Practices:**
        1. Use realistic student profiles for predictions
        2. Consider the model as a guidance tool, not absolute truth
        3. Combine with other evaluation criteria
        4. Update model periodically with new admission data
        5. Validate predictions against actual admission outcomes
        
        **Common Mistakes to Avoid:**
        - ❌ Entering scores outside valid ranges
        - ❌ Treating predictions as guaranteed outcomes
        - ❌ Ignoring other important admission factors
        - ❌ Using outdated model without retraining
        """
        st.markdown(tips)

if __name__ == "__main__":
    show()