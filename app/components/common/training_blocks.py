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

def render_linear_regression_parameters_block(page_key: str):
    st.markdown("### Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fit_intercept = st.checkbox(
            "Fit Intercept",
            value=True,
            help="Whether to calculate the intercept for this model",
            key=f"{page_key}_fit_intercept"
        )
        positive = st.checkbox(
            "Positive Coefficients",
            value=False,
            help="Force coefficients to be positive",
            key=f"{page_key}_positive"
        )
    
    with col2:
        copy_X = st.checkbox(
            "Copy X",
            value=True,
            help="If True, X will be copied; else, it may be overwritten",
            key=f"{page_key}_copy_X"
        )
    
    params = {
        "fit_intercept": fit_intercept,
        "positive": positive,
        "copy_X": copy_X,
    }
    
    return params

def render_logistic_regression_parameters_block(page_key: str):
    """Render parameter controls for Logistic Regression."""
    st.markdown("### Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        penalty = st.selectbox(
            "Penalty",
            ["l2", "l1", "elasticnet", None],
            index=0,
            help="Regularization penalty type",
            key=f"{page_key}_penalty"
        )
        C = st.number_input(
            "C (Inverse Regularization Strength)",
            min_value=0.001,
            max_value=100.0,
            value=1.0,
            format="%.3f",
            help="Smaller values specify stronger regularization",
            key=f"{page_key}_C"
        )
        solver = st.selectbox(
            "Solver",
            ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
            index=0,
            help="Algorithm to use in the optimization problem",
            key=f"{page_key}_solver"
        )
        max_iter = st.slider(
            "Max Iterations",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Maximum number of iterations for solver to converge",
            key=f"{page_key}_max_iter"
        )
    
    with col2:
        fit_intercept = st.checkbox(
            "Fit Intercept",
            value=True,
            help="Whether to calculate the intercept for this model",
            key=f"{page_key}_fit_intercept"
        )
        class_weight = st.selectbox(
            "Class Weight",
            [None, "balanced"],
            index=0,
            help="Weights associated with classes. 'balanced' adjusts weights inversely proportional to class frequencies",
            key=f"{page_key}_class_weight"
        )
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=1000,
            value=42,
            help="Controls randomness of the estimator",
            key=f"{page_key}_lr_random_state"
        )
        
        # Show l1_ratio only if elasticnet is selected
        if penalty == "elasticnet":
            l1_ratio = st.slider(
                "L1 Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="The Elastic-Net mixing parameter (0 = L2, 1 = L1)",
                key=f"{page_key}_l1_ratio"
            )
        else:
            l1_ratio = None
    
    # Build params dictionary
    params = {
        "penalty": penalty,
        "C": C,
        "solver": solver,
        "max_iter": max_iter,
        "fit_intercept": fit_intercept,
        "class_weight": class_weight,
        "random_state": random_state,
    }
    
    # Add l1_ratio only if elasticnet is selected
    if l1_ratio is not None:
        params["l1_ratio"] = l1_ratio
    
    return params

def render_random_forest_parameters_block(page_key: str, task_type: str):
    st.markdown("### Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider(
            "Number of Trees",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="The number of trees in the forest",
            key=f"{page_key}_n_estimators"
        )
        max_depth = st.slider(
            "Max Depth",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Maximum depth of the tree (None for unlimited)",
            key=f"{page_key}_max_depth"
        )
        min_samples_split = st.slider(
            "Min Samples Split",
            min_value=2,
            max_value=20,
            value=2,
            step=1,
            help="Minimum number of samples required to split an internal node",
            key=f"{page_key}_min_samples_split"
        )
        min_samples_leaf = st.slider(
            "Min Samples Leaf",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help="Minimum number of samples required to be at a leaf node",
            key=f"{page_key}_min_samples_leaf"
        )
    
    with col2:
        max_features = st.selectbox(
            "Max Features",
            ["sqrt", "log2", None],
            index=0,
            help="Number of features to consider when looking for the best split",
            key=f"{page_key}_max_features"
        )
        bootstrap = st.checkbox(
            "Bootstrap",
            value=True,
            help="Whether bootstrap samples are used when building trees",
            key=f"{page_key}_bootstrap"
        )
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=1000,
            value=42,
            help="Controls randomness of the estimator",
            key=f"{page_key}_rf_random_state"
        )
        
        if task_type == "classification":
            criterion = st.selectbox(
                "Criterion",
                ["gini", "entropy", "log_loss"],
                help="Function to measure the quality of a split",
                key=f"{page_key}_criterion"
            )
        else:  # regression
            criterion = st.selectbox(
                "Criterion",
                ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                help="Function to measure the quality of a split",
                key=f"{page_key}_criterion"
            )
    
    if max_depth == 50:
        use_max_depth = st.checkbox(
            "Use unlimited depth",
            value=False,
            key=f"{page_key}_unlimited_depth"
        )
        if use_max_depth:
            max_depth = None
    
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "random_state": random_state,
        "criterion": criterion,
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
    
    if model_name == "Neural Network":
        parameters = render_neural_networks_parameters_block(page_key=f"{page_key}_parameters")
    elif model_name == "Linear Regression":
        parameters = render_linear_regression_parameters_block(page_key=f"{page_key}_parameters")
    elif model_name == "Logistic Regression":
        parameters = render_logistic_regression_parameters_block(page_key=f"{page_key}_parameters")
    elif model_name in ["Random Forest", "Random Forest Regressor"]:
        parameters = render_random_forest_parameters_block(page_key=f"{page_key}_parameters", task_type=task_type)

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
