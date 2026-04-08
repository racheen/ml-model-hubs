import streamlit as st
import pandas as pd
from app.components.common.display import get_available_models, plot_clusters
from app.components.common.forms import render_raw_input_form
from app.components.common.layout import render_section_title
from app.config.paths import DATA_DIR
from app.core.state_manager import set_page_value
from app.domain.clustering.config import CATEGORY, DEFAULT_MODELS, RAW_FORM_FIELDS, USE_JOBLIB
from app.domain.clustering.preprocess import preprocess_input
from app.repositories.model_repository import load_model, load_scaler

def render_predict_section():
    render_section_title("Predict with Default Model", "Assign a sample customer to a saved cluster.")
    available_models = get_available_models(CATEGORY, USE_JOBLIB)
    
    if not available_models:
        st.warning(f"No models found in category '{CATEGORY}'. Using default models list.")
        available_models = DEFAULT_MODELS
    
    model_name = st.selectbox(
        "Models", 
        available_models, 
        key="clustering_default_model",
        help=f"Select from {len(available_models)} available model(s)"
    )
    submitted, raw_payload = render_raw_input_form("clustering_predict_form", RAW_FORM_FIELDS)
    if submitted:
        try:
            processed = preprocess_input(raw_payload)
            model = load_model(model_name=model_name, category=CATEGORY, use_joblib=USE_JOBLIB)
            if model is None:
                raise FileNotFoundError("KMeans model file not found.")
            scaler = load_scaler(CATEGORY)
            fit_input = scaler.transform(processed) if scaler is not None else processed
            cluster = int(model.predict(fit_input)[0])
            set_page_value("clustering", "default_prediction", cluster)
            set_page_value("clustering", "default_model_name", model_name)
            st.success(f"Assigned cluster: {cluster}")
            
            st.markdown("### Your Position in the Cluster Map")
            df = pd.read_csv(f'{DATA_DIR}/mall_customers.csv')
            st.pyplot(plot_clusters(df, model, user_point=(processed["Annual_Income"], processed["Spending_Score"])))

            if hasattr(model, "cluster_centers_"):
                st.markdown("**Cluster centers**")
                st.dataframe(model.cluster_centers_)
            st.markdown("**Processed feature vector**")
            st.dataframe(processed, use_container_width=True)
        except Exception as exc:
            st.error(f"Cluster prediction failed: {exc}")
