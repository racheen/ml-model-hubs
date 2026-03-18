import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.model_manager import load_model, get_model_info

def show():
    """Display clustering/segmentation page"""
    st.header("Customer Segmentation (K-Means Clustering)")
    
    st.markdown("""
    This model segments mall customers into groups based on:
    - **Age**
    - **Annual Income**
    - **Spending Score**
    """)
    
    # Load the clustering model
    try:
        model = load_model('KMeans', category='Clustering')
        if model is None:
            st.error("Clustering model not found. Please train it first.")
            st.stop()
        st.success("K-Means model loaded!")
    except TypeError:
        # Fallback if category parameter not supported
        model = load_model('KMeans')
        if model is None:
            st.error("Clustering model not found. Please train it first.")
            st.stop()
        st.success("K-Means model loaded!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Display model info
    model_info = get_model_info('KMeans', 'Clustering')
    if model_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Clusters", model.n_clusters)
        with col2:
            st.metric("Model Size", f"{model_info['size_mb']:.2f} MB")
        with col3:
            st.metric("Initialization", "K-Means++")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Single Customer", "Batch Upload", "Cluster Analysis"])
    
    with tab1:
        st.markdown("### Predict Single Customer Segment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
        with col2:
            income = st.number_input("Annual Income ($1000s)", min_value=15, max_value=150, value=50)
        with col3:
            spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
        
        if st.button("🔮 Predict Segment", use_container_width=True):
            input_data = np.array([[age, income, spending_score]])
            
            try:
                cluster = model.predict(input_data)[0]
                
                st.markdown("---")
                st.markdown("### Segment Result")
                
                # Color code the clusters
                colors = ['🔴', '🟠', '🟡', '🟢', '🔵']
                color = colors[cluster % len(colors)]
                
                st.metric("Customer Segment", f"{color} Cluster {cluster}")
                
                # Cluster descriptions (customize based on your analysis)
                cluster_descriptions = {
                    0: "Budget-Conscious Shoppers - Low income, low spending",
                    1: "Regular Customers - Medium income, medium spending",
                    2: "Premium Customers - High income, high spending",
                    3: "Young Professionals - Medium income, high spending",
                    4: "Senior Customers - High income, low spending"
                }
                
                if cluster in cluster_descriptions:
                    st.info(f"**Profile:** {cluster_descriptions[cluster]}")
                
                # Display customer profile
                st.markdown("### Customer Profile")
                profile = f"""
                - **Age:** {age} years
                - **Annual Income:** ${income}k
                - **Spending Score:** {spending_score}/100
                """
                st.markdown(profile)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with tab2:
        st.markdown("### Segment Multiple Customers")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                
                # Check required columns
                required_cols = ['Age', 'Annual_Income', 'Spending_Score']
                if not all(col in df_upload.columns for col in required_cols):
                    st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                    st.stop()
                
                # Make predictions
                X = df_upload[required_cols].values
                clusters = model.predict(X)
                df_upload['Cluster'] = clusters
                
                st.success(f"Segmented {len(df_upload)} customers!")
                
                # Display results
                st.dataframe(df_upload, use_container_width=True)
                
                # Download results
                csv = df_upload.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="customer_segments.csv",
                    mime="text/csv"
                )
                
                # Visualize clusters
                st.markdown("### Cluster Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                df_upload['Cluster'].value_counts().sort_index().plot(kind='bar', ax=ax, color='steelblue')
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Number of Customers')
                ax.set_title('Customer Distribution by Segment')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.markdown("### Cluster Analysis")
        
        # Load sample data for analysis
        try:
            from pathlib import Path
            
            # Get the project root directory
            project_root = Path(__file__).parent.parent
            data_path = project_root / 'data' / 'mall_customers.csv'
            
            df_sample = pd.read_csv(data_path)
            
            # Get cluster assignments
            X_sample = df_sample[['Age', 'Annual_Income', 'Spending_Score']].values
            df_sample['Cluster'] = model.predict(X_sample)
            
            # Cluster statistics
            st.markdown("#### Cluster Statistics")
            cluster_stats = df_sample.groupby('Cluster')[['Age', 'Annual_Income', 'Spending_Score']].agg(['mean', 'min', 'max'])
            st.dataframe(cluster_stats, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Income vs Spending Score")
                fig, ax = plt.subplots(figsize=(8, 6))
                for cluster in df_sample['Cluster'].unique():
                    cluster_data = df_sample[df_sample['Cluster'] == cluster]
                    ax.scatter(cluster_data['Annual_Income'], cluster_data['Spending_Score'], 
                             label=f'Cluster {cluster}', s=50, alpha=0.6)
                
                # Plot cluster centers
                centers = model.cluster_centers_
                ax.scatter(centers[:, 1], centers[:, 2], c='black', s=300, alpha=0.8, marker='X', label='Centroids')
                
                ax.set_xlabel('Annual Income ($1000s)')
                ax.set_ylabel('Spending Score')
                ax.set_title('Customer Segments')
                ax.legend()
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### Age Distribution by Cluster")
                fig, ax = plt.subplots(figsize=(8, 6))
                df_sample.boxplot(column='Age', by='Cluster', ax=ax)
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Age')
                ax.set_title('Age Distribution by Segment')
                plt.suptitle('')
                st.pyplot(fig)
        
        except FileNotFoundError:
            st.warning("Sample data file not found for analysis")
        except Exception as e:
            st.error(f"Error in cluster analysis: {str(e)}")

if __name__ == "__main__":
    show()
