import streamlit as st
from app.components.common.forms import render_raw_input_form
from app.components.common.display import get_available_models, show_probability
from app.components.common.layout import render_section_title
from app.core.state_manager import set_page_value
from app.services.prediction_service import predict_with_default_model
from app.domain.student_admission.config import CATEGORY, DEFAULT_MODELS, RAW_FORM_FIELDS, USE_JOBLIB
from app.domain.student_admission.preprocess import preprocess_input

def render_predict_section():
    render_section_title("Predict with Default Model", "Choose a pretrained model and enter one sample.")
    available_models = get_available_models(CATEGORY, USE_JOBLIB)
    
    if not available_models:
        st.warning(f"No models found in category '{CATEGORY}'. Using default models list.")
        available_models = DEFAULT_MODELS
    
    model_name = st.selectbox(
        "Models", 
        available_models, 
        key="neural_networks_default_model", 
        help=f"Select from {len(available_models)} available model(s)")
    
    submitted, raw_payload = render_raw_input_form("neural_networks_predict_form", RAW_FORM_FIELDS, 2)
    if submitted:
        try:
            processed = preprocess_input(raw_payload)
            result = predict_with_default_model(
                category=CATEGORY,
                model_name=model_name,
                processed_input=processed,
                use_joblib=USE_JOBLIB,
            )
            set_page_value("neural_networks", "default_prediction", result.prediction)
            set_page_value("neural_networks", "default_model_name", model_name)
            
            if result.prediction == 1:
                st.success(f"**LIKELY TO BE ADMITTED**")
            else:
                st.error(f"**UNLIKELY TO BE ADMITTED**")
            show_probability(result.probabilities, "Admission Probability")
            st.markdown("**Processed feature vector**")
            st.dataframe(processed, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
