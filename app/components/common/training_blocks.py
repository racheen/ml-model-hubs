import pandas as pd
import streamlit as st

from app.components.common.display import show_metric_cards, show_dataframe
from app.config.paths import DATA_DIR
from app.repositories.model_repository import save_feature_names, save_model, save_scaler
from app.services.training_service import train_supervised_model, train_clustering_model
from app.core.state_manager import set_page_value

def render_neural_networks_parameters_block(page_key: str):
    st.markdown("### Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hidden_layer_sizes = st.text_input(
            "Hidden Layer Sizes",
            value="100",
            help="Comma-separated values for hidden layer sizes (e.g., '100' or '100,50')",
            key=f"{page_key}_hidden_layers"
        )
        activation = st.selectbox(
            "Activation Function",
            ["relu", "tanh", "logistic", "identity"],
            key=f"{page_key}_activation"
        )
        solver = st.selectbox(
            "Solver",
            ["adam", "sgd", "lbfgs"],
            key=f"{page_key}_solver"
        )
        alpha = st.number_input(
            "Alpha (L2 penalty)",
            min_value=0.0001,
            max_value=1.0,
            value=0.0001,
            format="%.4f",
            key=f"{page_key}_alpha"
        )
    
    with col2:
        learning_rate = st.selectbox(
            "Learning Rate",
            ["constant", "invscaling", "adaptive"],
            key=f"{page_key}_learning_rate"
        )
        learning_rate_init = st.number_input(
            "Initial Learning Rate",
            min_value=0.0001,
            max_value=1.0,
            value=0.001,
            format="%.4f",
            key=f"{page_key}_learning_rate_init"
        )
        max_iter = st.slider(
            "Max Iterations",
            min_value=100,
            max_value=1000,
            value=200,
            step=50,
            key=f"{page_key}_max_iter"
        )
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=1000,
            value=42,
            key=f"{page_key}_random_state"
        )
    
    
    try:
        if ',' in hidden_layer_sizes:
            hidden_layers_tuple = tuple(int(x.strip()) for x in hidden_layer_sizes.split(','))
        else:
            hidden_layers_tuple = (int(hidden_layer_sizes.strip()),)
    except ValueError:
        st.error("Invalid hidden layer sizes. Using default (100,)")
        hidden_layers_tuple = (100,)
    
    # Return parameters as a dictionary
    params = {
        "hidden_layer_sizes": hidden_layers_tuple,
        "activation": activation,
        "solver": solver,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "learning_rate_init": learning_rate_init,
        "max_iter": max_iter,
        "random_state": random_state,
    }
    
    return params
def render_supervised_training_block(page_key: str, category: str, task_type: str, target_column: str, training_models: list[str], csv_file: str = "Admission.csv" ):
    df = pd.read_csv(f"{DATA_DIR}/{csv_file}")

    if target_column not in df.columns:
        target_column = st.selectbox("Target column", df.columns.tolist(), key=f"{page_key}_target_column")
    else:
        st.success(f"Using detected target column: `{target_column}`")

    model_name = st.selectbox("Training algorithm", training_models, key=f"{page_key}_training_model")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key=f"{page_key}_test_size")
    scale_features = st.checkbox("Scale features before training", value=True, key=f"{page_key}_scale_features")
    save_artifacts = st.checkbox("Save custom model to models folder", value=True, key=f"{page_key}_save_model")
    
    if model_name in ["Neural Network"]: 
        parameters = render_neural_networks_parameters_block(page_key=f"{page_key}_parameters")

    if st.button("Train Custom Model", key=f"{page_key}_train_button"):
        with st.spinner("Training model..."):
            result = train_supervised_model(
                df=df,
                target_column=target_column,
                task_type=task_type,
                model_name=model_name,
                scale_features=scale_features,
                test_size=test_size,
                model_params=parameters if model_name == "Neural Network" else None,
            )
        set_page_value(page_key, "custom_training_result", result)
        st.success("Custom model trained successfully.")
        show_metric_cards(result.metrics)

        if save_artifacts:
            save_model(result.estimator, f"Custom {model_name}", category=category, use_joblib=False)
            save_feature_names(result.feature_names, category=category)
            if result.scaler is not None:
                save_scaler(result.scaler, category=category)
            st.caption("Saved custom model artifacts into the category folder.")

def render_clustering_training_block(page_key: str, category: str, expected_features: list[str]):
    n_clusters = st.slider("Number of clusters", 2, 10, 5)
    init = st.selectbox("Initialization", ["k-means++", "random"])
    max_iter = st.slider("Max iterations", 100, 1000, 300)
    n_init = st.slider("n_init", 5, 30, 10)
    scale_features = st.checkbox("Scale features", True)

    df = pd.read_csv(f"{DATA_DIR}/mall_customers.csv")

    defaults = [c for c in expected_features if c in df.columns]
    selected_features = st.multiselect("Feature columns", df.columns.tolist(), default=defaults, key=f"{page_key}_cluster_features")
    n_clusters = st.slider("Number of clusters", 2, 10, 5, 1, key=f"{page_key}_n_clusters")
    scale_features = st.checkbox("Scale features before clustering", value=True, key=f"{page_key}_cluster_scale")
    save_artifacts = st.checkbox("Save custom cluster model", value=True, key=f"{page_key}_cluster_save")

    if st.button("Train Clustering Model", key=f"{page_key}_cluster_train_button"):
        if not selected_features:
            st.error("Select at least one feature.")
            return
        with st.spinner("Training KMeans..."):
            result = train_clustering_model(
                df=df,
                feature_columns=selected_features,
                n_clusters=n_clusters,
                scale_features=scale_features,
                init=init,
                max_iter=max_iter,
                n_init=n_init,
            )
        set_page_value(page_key, "custom_training_result", result)
        st.success("Custom clustering model trained successfully.")
        show_metric_cards(result.metrics)

        if save_artifacts:
            save_model(result.estimator, "Custom KMeans", category=category, use_joblib=False)
            save_feature_names(result.feature_names, category=category)
            if result.scaler is not None:
                save_scaler(result.scaler, category=category)
            st.caption("Saved custom clustering artifacts into the category folder.")
