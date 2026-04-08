import streamlit as st
from app.services.evaluation_service import feature_importance_frame
from app.repositories.model_repository import load_model, get_model_info, load_metrics

def render_default_model_analysis(category: str, model_name: str, feature_names: list[str] | None, use_joblib: bool = False):
    info = get_model_info(model_name=model_name, category=category, use_joblib=use_joblib)
    model = load_model(model_name=model_name, category=category, use_joblib=use_joblib)
    
    if info is None:
        st.warning("Default model file was not found.")
        return
    
    if info:
        
        st.metric("Model Type", info["name"])

        st.metric("Estimator Type", info["type"])
    
        st.metric("Size (MB)", info["size_mb"])

        if hasattr(model, 'n_clusters'):
            st.metric("Number of Clusters", model.n_clusters)
        
        if hasattr(model, 'hidden_layer_sizes'):
            hidden_layers = model.hidden_layer_sizes
            if isinstance(hidden_layers, tuple):
                layer_info = f"{len(hidden_layers)} layer(s): {hidden_layers}"
            else:
                layer_info = f"1 layer: ({hidden_layers},)"
            st.metric("Hidden Layers", layer_info)

        if hasattr(model, 'intercept_'):
            st.metric("Intercepts", model.intercept_)
        
        if hasattr(model, 'n_estimators'):
            st.metric("Tree", model.n_estimators)


    if model is not None and feature_names:
        frame = feature_importance_frame(model, feature_names)
        if frame is not None:
            st.markdown("**Model interpretation**")
            st.dataframe(frame, use_container_width=True, height=320)
        else:
            st.info("This model does not expose feature importances or coefficients.")
    

    metrics = load_metrics(model_name=model_name, category=category)
    if metrics:
        st.markdown("**Model Performance:**")
                     
        # R² Score
        r2 = metrics.get('r2_score', None)
        if r2 is not None:
            st.metric("R² Score", f"{r2:.4f}", 
                        help="Coefficient of determination - measures how well predictions fit actual values")
        
        # Error Metrics
        mae = metrics.get('mae', None)
        if mae is not None:
            st.metric("MAE", f"${mae:,.2f}", 
                        help="Mean Absolute Error - average prediction error")
        
        rmse = metrics.get('rmse', None)
        if rmse is not None:
            st.metric("RMSE", f"${rmse:,.2f}", 
                        help="Root Mean Squared Error - penalizes larger errors")
    
    
        mse = metrics.get('mse', None)
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
            f_stat = metrics.get('f_statistic', None)
            if f_stat is not None:
                st.metric("F-Statistic", f"{f_stat:.2f}", 
                            help="Tests if model is statistically significant")
        
        with stat_col2:
            p_val = metrics.get('p_value', None)
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
                
