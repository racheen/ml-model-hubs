import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.model_manager import load_model, get_model_info
import joblib
import os

def show():
    """Display real estate pricing prediction page"""
    st.header("Real Estate Price Prediction")
    
    st.markdown("""
    This model predicts property prices based on:
    - **Year Built**
    - **Square Footage**
    - **Number of Bedrooms**
    - **Property Type**
    - **And more features...**
    """)
    
    # Load the real estate models
    try:
        lr_model = load_model('Linear_Regression', category='Real Estate')
        rf_model = load_model('Random_Forest', category='Real Estate')
        
        if lr_model is None and rf_model is None:
            st.error("Real Estate models not found. Please train them first.")
            st.stop()
        
        st.success("Real Estate models loaded!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()
    
    # Display model info
    col1, col2 = st.columns(2)
    
    with col1:
        if lr_model:
            lr_info = get_model_info('Linear_Regression', 'Real Estate')
            if lr_info:
                st.metric("Linear Regression Model", f"{lr_info['size_mb']:.2f} MB")
    
    with col2:
        if rf_model:
            rf_info = get_model_info('Random_Forest', 'Real Estate')
            if rf_info:
                st.metric("Random Forest Model", f"{rf_info['size_mb']:.2f} MB")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Single Property", "Batch Upload", "Model Comparison"])
    
    with tab1:
        st.markdown("### Predict Single Property Price")
        
        # Create input columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year_sold = st.number_input("Year Sold", min_value=1990, max_value=2024, value=2024)
        with col2:
            property_tax = st.number_input("Property Tax ($)", min_value=50, max_value=1000, value=250)
        with col3:
            insurance = st.number_input("Insurance ($)", min_value=20, max_value=500, value=80)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
        with col5:
            bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
        with col6:
            sqft = st.number_input("Square Footage", min_value=500, max_value=10000, value=2000)
        
        col7, col8, col9 = st.columns(3)
        
        with col7:
            year_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=2010)
        with col8:
            lot_size = st.number_input("Lot Size (sqft)", min_value=0, max_value=500000, value=5000)
        with col9:
            basement = st.selectbox("Has Basement?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        col10, col11, col12 = st.columns(3)
        
        with col10:
            popular = st.selectbox("Popular Area?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with col11:
            recession = st.selectbox("During Recession?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with col12:
            property_type = st.selectbox("Property Type", ["House", "Condo", "Townhouse", "Land"])
        
        col13, = st.columns(1)

        with col13:
            property_age = st.number_input("Property Age (years)", min_value=0, max_value=200, value=year_sold - year_built)
        
        # Calculate property age
        property_type_condo = 1 if property_type == "Condo" else 0
        
        # Model selection
        st.markdown("---")
        model_choice = st.radio("Select Model for Prediction", 
                               ["Linear Regression", "Random Forest", "Compare Both"],
                               horizontal=True)
        
        if st.button("Predict Price", use_container_width=True):
            # Prepare input data with all required features from training data
            input_data = np.array([[
                year_sold,
                property_tax,
                insurance,
                bedrooms,
                bathrooms,
                sqft,
                year_built,
                lot_size,
                basement,
                popular,
                recession,
                property_age,
                property_type_condo
            ]])

            try:
                st.markdown("---")
                st.markdown("### Price Prediction Result")
                
                lr_price = None
                rf_price = None
                
                if model_choice in ["Linear Regression", "Compare Both"] and lr_model:
                    lr_price = lr_model.predict(input_data)[0]
                    st.metric("Linear Regression Prediction", f"${lr_price:,.2f}")
                
                if model_choice in ["Random Forest", "Compare Both"] and rf_model:
                    rf_price = rf_model.predict(input_data)[0]
                    st.metric("Random Forest Prediction", f"${rf_price:,.2f}")
                
                if model_choice == "Compare Both" and lr_price is not None and rf_price is not None:
                    avg_price = (lr_price + rf_price) / 2
                    st.info(f"**Average Prediction:** ${avg_price:,.2f}")
                    
                    # Show difference
                    difference = abs(lr_price - rf_price)
                    st.warning(f"**Model Difference:** ${difference:,.2f}")
                
                # Display property details
                st.markdown("### Property Details")
                details = f"""
                - **Year Sold:** {year_sold}
                - **Property Tax:** ${property_tax}
                - **Insurance:** ${insurance}
                - **Bedrooms:** {bedrooms}
                - **Bathrooms:** {bathrooms}
                - **Square Footage:** {sqft:,} sqft
                - **Year Built:** {year_built}
                - **Property Age:** {property_age} years
                - **Lot Size:** {lot_size:,} sqft
                - **Has Basement:** {"Yes" if basement == 1 else "No"}
                - **Popular Area:** {"Yes" if popular == 1 else "No"}
                - **During Recession:** {"Yes" if recession == 1 else "No"}
                - **Property Type:** {property_type}
                """
                st.markdown(details)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with tab2:
        st.markdown("### Predict Multiple Properties")
        
        uploaded_file = st.file_uploader("Upload CSV file with property data", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                
                # Check required columns
                required_cols = ['year_sold', 'property_tax', 'insurance', 'beds', 'baths', 'sqft', 
                               'year_built', 'lot_size', 'basement', 'popular', 'recession', 'property_age', 'property_type_Condo']
                missing_cols = [col for col in required_cols if col not in df_upload.columns]
                
                if missing_cols:
                    st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                    st.stop()
                
                # Select model for batch prediction
                batch_model = st.radio("Select Model", 
                                      ["Linear Regression", "Random Forest"],
                                      horizontal=True)
                
                if st.button("Predict Prices", use_container_width=True):
                    X = df_upload[required_cols].values
                    
                    if batch_model == "Linear Regression" and lr_model:
                        predictions = lr_model.predict(X)
                        model_name = "Linear Regression"
                    else:
                        predictions = rf_model.predict(X)
                        model_name = "Random Forest"
                    
                    df_upload['Predicted_Price'] = predictions
                    
                    st.success(f"Predicted prices for {len(df_upload)} properties using {model_name}!")
                    
                    # Display results
                    st.dataframe(df_upload, use_container_width=True)
                    
                    # Download results
                    csv = df_upload.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="property_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Statistics
                    st.markdown("### Prediction Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Average Price", f"${predictions.mean():,.2f}")
                    with col2:
                        st.metric("Min Price", f"${predictions.min():,.2f}")
                    with col3:
                        st.metric("Max Price", f"${predictions.max():,.2f}")
                    with col4:
                        st.metric("Std Dev", f"${predictions.std():,.2f}")
                    
                    # Visualization
                    st.markdown("### Price Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(predictions, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                    ax.set_xlabel('Predicted Price ($)')
                    ax.set_ylabel('Number of Properties')
                    ax.set_title('Distribution of Predicted Prices')
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.markdown("### Model Comparison & Analysis")
        
        if lr_model and rf_model:
            st.info("Comparing Linear Regression vs Random Forest models")
            
            # Model information with detailed statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Linear Regression")
                lr_info = get_model_info('Linear_Regression', 'Real Estate')
                if lr_info:
                    st.write(f"**Type:** {lr_info['type']}")
                    st.write(f"**Size:** {lr_info['size_mb']:.2f} MB")
                
                # Display model coefficients and statistics
                st.markdown("**Model Statistics:**")
                try:
                    # Get feature names
                    feature_names = ['year_sold', 'property_tax', 'insurance', 'beds', 'baths', 
                                   'sqft', 'year_built', 'lot_size', 'basement', 'popular', 
                                   'recession', 'property_age', 'property_type_Condo']
                    
                    # Display coefficients
                    coef_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': lr_model.coef_
                    }).sort_values('Coefficient', key=abs, ascending=False)
                    
                    st.dataframe(coef_df, use_container_width=True, height=300)
                    
                    # Display intercept
                    st.metric("Intercept", f"{lr_model.intercept_:,.2f}")
                    
                    # Calculate R² on training data if available
                    # Note: You'll need to load training data or save these metrics during training
                    st.info("💡 Tip: Train the model with cross-validation to see R², MSE, and F-statistics")
                    
                except Exception as e:
                    st.warning(f"Could not display detailed statistics: {str(e)}")
                
                st.write("**Pros:**")
                st.write("- Fast predictions")
                st.write("- Interpretable coefficients")
                st.write("- Low memory usage")
                st.write("**Cons:**")
                st.write("- Assumes linear relationships")
                st.write("- May underfit complex data")
            
            with col2:
                st.markdown("#### Random Forest")
                rf_info = get_model_info('Random_Forest', 'Real Estate')
                if rf_info:
                    st.write(f"**Type:** {rf_info['type']}")
                    st.write(f"**Size:** {rf_info['size_mb']:.2f} MB")
                
                # Display Random Forest statistics
                st.markdown("**Model Statistics:**")
                try:
                    # Get feature names
                    feature_names = ['year_sold', 'property_tax', 'insurance', 'beds', 'baths', 
                                   'sqft', 'year_built', 'lot_size', 'basement', 'popular', 
                                   'recession', 'property_age', 'property_type_Condo']
                    
                    # Display feature importances
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': rf_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.dataframe(importance_df, use_container_width=True, height=300)
                    
                    # Display number of estimators
                    st.metric("Number of Trees", rf_model.n_estimators)
                    
                    # Feature importance visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(importance_df['Feature'], importance_df['Importance'], color='forestgreen', alpha=0.7)
                    ax.set_xlabel('Feature Importance')
                    ax.set_title('Random Forest Feature Importances')
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.warning(f"Could not display detailed statistics: {str(e)}")
                
                st.write("**Pros:**")
                st.write("- Captures non-linear patterns")
                st.write("- Robust to outliers")
                st.write("- Feature importance available")
                st.write("**Cons:**")
                st.write("- Slower predictions")
                st.write("- Higher memory usage")
            
            
            st.markdown("---")
            st.markdown("### Detailed Model Performance Metrics")
            
            try:
                lr_metrics_path = 'models/Real Estate/Linear_Regression_metrics.pkl'
                rf_metrics_path = 'models/Real Estate/Random_Forest_metrics.pkl'
                
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.markdown("#### Linear Regression Metrics")
                    if os.path.exists(lr_metrics_path):
                        lr_metrics = joblib.load(lr_metrics_path)
                        
                        st.markdown("**Model Performance:**")
                        
                        # R² Score
                        r2 = lr_metrics.get('r2_score', None)
                        if r2 is not None:
                            st.metric("R² Score", f"{r2:.4f}", 
                                     help="Coefficient of determination - measures how well predictions fit actual values")
                        
                        # Error Metrics
                        mae = lr_metrics.get('mae', None)
                        if mae is not None:
                            st.metric("MAE", f"${mae:,.2f}", 
                                        help="Mean Absolute Error - average prediction error")
                        
                        rmse = lr_metrics.get('rmse', None)
                        if rmse is not None:
                            st.metric("RMSE", f"${rmse:,.2f}", 
                                        help="Root Mean Squared Error - penalizes larger errors")
                    
                    
                        mse = lr_metrics.get('mse', None)
                        if mse is not None:
                            # Format MSE in scientific notation if very large
                            if mse > 1000000:
                                st.metric("MSE", f"${mse:,.0f}", 
                                            help="Mean Squared Error")
                            else:
                                st.metric("MSE", f"${mse:,.2f}", 
                                            help="Mean Squared Error")
                    
                    
                        st.markdown("**Statistical Significance:**")
                        stat_col1, stat_col2 = st.columns(2)
                        
                        with stat_col1:
                            f_stat = lr_metrics.get('f_statistic', None)
                            if f_stat is not None:
                                st.metric("F-Statistic", f"{f_stat:.2f}", 
                                         help="Tests if model is statistically significant")
                        
                        with stat_col2:
                            p_val = lr_metrics.get('p_value', None)
                            if p_val is not None:
                                if p_val < 0.0001:
                                    st.metric("P-Value", f"{p_val:.2e}", 
                                             help="Probability that results occurred by chance")
                                else:
                                    st.metric("P-Value", f"{p_val:.4f}", 
                                             help="Probability that results occurred by chance")
                        
                        if p_val is not None and p_val < 0.05:
                            st.success("Model is statistically significant (p < 0.05)")
                        elif p_val is not None:
                            st.warning("Model may not be statistically significant")
                            
                    else:
                        st.info("Metrics not saved. Retrain the model to see detailed statistics.")
                
                with metrics_col2:
                    st.markdown("#### Random Forest Metrics")
                    if os.path.exists(rf_metrics_path):
                        rf_metrics = joblib.load(rf_metrics_path)
                        
                        # Create a clean metrics display
                        st.markdown("**Model Performance:**")
                        
                        # R² Score
                        r2 = rf_metrics.get('r2_score', None)
                        if r2 is not None:
                            st.metric("R² Score", f"{r2:.4f}", 
                                     help="Coefficient of determination - measures how well predictions fit actual values")
                        
                        # Error Metrics
                        mae = rf_metrics.get('mae', None)
                        if mae is not None:
                            st.metric("MAE", f"${mae:,.2f}", 
                                        help="Mean Absolute Error - average prediction error")
                        
                        rmse = rf_metrics.get('rmse', None)
                        if rmse is not None:
                            st.metric("RMSE", f"${rmse:,.2f}", 
                                        help="Root Mean Squared Error - penalizes larger errors")
                    
                        mse = rf_metrics.get('mse', None)
                        if mse is not None:
                            # Format MSE in scientific notation if very large
                            if mse > 1000000:
                                st.metric("MSE", f"${mse:,.0f}", 
                                            help="Mean Squared Error")
                            else:
                                st.metric("MSE", f"${mse:,.2f}", 
                                            help="Mean Squared Error")
                        
                        # Random Forest Specific Metrics
                        st.markdown("**Random Forest Specific:**")
                        oob_col1, oob_col2 = st.columns(2)
                        
                        with oob_col1:
                            oob = rf_metrics.get('oob_score', None)
                            if oob is not None:
                                st.metric("OOB Score", f"{oob:.4f}", 
                                         help="Out-of-Bag score - internal validation metric")
                        
                        with oob_col2:
                            n_trees = rf_metrics.get('n_estimators', None)
                            if n_trees is not None:
                                st.metric("Trees", f"{n_trees}", 
                                         help="Number of decision trees in the forest")
                        
                        # Performance indicator
                        if r2 is not None:
                            if r2 > 0.8:
                                st.success("Excellent model performance (R² > 0.8)")
                            elif r2 > 0.6:
                                st.info("✓ Good model performance (R² > 0.6)")
                            else:
                                st.warning("Model performance could be improved")
                                
                    else:
                        st.info("Metrics not saved. Retrain the model to see detailed statistics.")
                
                # Model Comparison Summary
                if os.path.exists(lr_metrics_path) and os.path.exists(rf_metrics_path):
                    st.markdown("---")
                    st.markdown("#### Model Comparison Summary")
                    
                    lr_metrics = joblib.load(lr_metrics_path)
                    rf_metrics = joblib.load(rf_metrics_path)
                    
                    comparison_data = {
                        'Metric': ['R² Score', 'MAE', 'RMSE', 'MSE'],
                        'Linear Regression': [
                            f"{lr_metrics.get('r2_score', 0):.4f}",
                            f"${lr_metrics.get('mae', 0):,.2f}",
                            f"${lr_metrics.get('rmse', 0):,.2f}",
                            f"${lr_metrics.get('mse', 0):,.0f}"
                        ],
                        'Random Forest': [
                            f"{rf_metrics.get('r2_score', 0):.4f}",
                            f"${rf_metrics.get('mae', 0):,.2f}",
                            f"${rf_metrics.get('rmse', 0):,.2f}",
                            f"${rf_metrics.get('mse', 0):,.0f}"
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Winner determination
                    lr_r2 = lr_metrics.get('r2_score', 0)
                    rf_r2 = rf_metrics.get('r2_score', 0)
                    
                    if rf_r2 > lr_r2:
                        improvement = ((rf_r2 - lr_r2) / lr_r2) * 100
                        st.success(f"**Random Forest** performs better with {improvement:.1f}% higher R² score")
                    elif lr_r2 > rf_r2:
                        improvement = ((lr_r2 - rf_r2) / rf_r2) * 100
                        st.success(f"**Linear Regression** performs better with {improvement:.1f}% higher R² score")
                    else:
                        st.info("⚖️ Both models perform similarly")
                           
            except Exception as e:
                st.warning(f"Could not load saved metrics: {str(e)}")
                st.info("💡 To see detailed metrics, update your training notebook to save metrics using joblib")
            
            # Comparison test
            st.markdown("---")
            st.markdown("#### Test Predictions on Sample Data")
            
            test_samples = np.array([
                [2013, 234, 81, 1, 1, 584, 2013, 0, 0, 0, 1, 0, 1],  # price, year_sold, property_tax, insurance, beds, baths, sqft, year_built, lot_size, basement, popular, recession, property_age, property_type_Condo
                [2012, 216, 74, 1, 1, 615, 1963, 0, 0, 0, 1, 49, 1],
                [2005, 265, 92, 1, 1, 618, 2000, 33541, 1, 0, 0, 5, 0],
                [2011, 176, 61, 1, 1, 642, 1944, 0, 0, 0, 1, 67, 1],
            ])
            
            lr_preds = lr_model.predict(test_samples)
            rf_preds = rf_model.predict(test_samples)
            
            comparison_df = pd.DataFrame({
                'Year': [2010, 2015, 2005, 2020],
                'Sqft': [2000, 3000, 1500, 2500],
                'Bedrooms': [3, 4, 2, 3],
                'Linear Regression': lr_preds,
                'Random Forest': rf_preds,
                'Difference': np.abs(lr_preds - rf_preds)
            })
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            x_pos = np.arange(len(test_samples))
            width = 0.35
            
            ax.bar(x_pos - width/2, lr_preds, width, label='Linear Regression', alpha=0.8)
            ax.bar(x_pos + width/2, rf_preds, width, label='Random Forest', alpha=0.8)
            
            ax.set_xlabel('Sample Property')
            ax.set_ylabel('Predicted Price ($)')
            ax.set_title('Model Prediction Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'Property {i+1}' for i in range(len(test_samples))])
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        
        else:
            st.warning("Both models are required for comparison. Please ensure both models are loaded.")

if __name__ == "__main__":
    show()

        