import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.model_manager import load_model, get_model_info

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
        
        # Calculate property age
        property_age = year_sold - year_built
        property_type_condo = 1 if property_type == "Condo" else 0
        
        # Model selection
        st.markdown("---")
        model_choice = st.radio("Select Model for Prediction", 
                               ["Linear Regression", "Random Forest", "Compare Both"],
                               horizontal=True)
        
        if st.button("🔮 Predict Price", use_container_width=True):
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
                
                if st.button("🔮 Predict Prices", use_container_width=True):
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
            
            # Model information
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Linear Regression")
                lr_info = get_model_info('Linear_Regression', 'Real Estate')
                if lr_info:
                    st.write(f"**Type:** {lr_info['type']}")
                    st.write(f"**Size:** {lr_info['size_mb']:.2f} MB")
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
                st.write("**Pros:**")
                st.write("- Captures non-linear patterns")
                st.write("- Robust to outliers")
                st.write("- Feature importance available")
                st.write("**Cons:**")
                st.write("- Slower predictions")
                st.write("- Higher memory usage")
            
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

        