import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans
from app.core.schemas import TrainingResult
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from app.core.schemas import TrainingResult
from app.services.evaluation_service import evaluate_supervised_model, evaluate_clustering_model
from app.utils.logger import get_logger

logger = get_logger(__name__)

def _get_supervised_model(task_type: str, model_name: str, parameters: dict = None):
    """Get supervised model instance based on task type and model name."""
    if parameters is None:
        parameters = {}
    
    if task_type == "classification":
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=1000, **parameters)
        elif model_name == "Random Forest":
            return RandomForestClassifier(random_state=42, **parameters)
        elif model_name == "Neural Network":
            return MLPClassifier(
                hidden_layer_sizes=parameters.get("hidden_layer_sizes", (32, 16)),
                max_iter=parameters.get("max_iter", 500),
                activation=parameters.get("activation", "relu"),
                solver=parameters.get("solver", "adam"),
                alpha=parameters.get("alpha", 0.0001),
                learning_rate=parameters.get("learning_rate", "constant"),
                learning_rate_init=parameters.get("learning_rate_init", 0.001),
                random_state=parameters.get("random_state", 42),
            )
        else:
            logger.error(f"Unknown classification model: {model_name}")
            raise ValueError(f"Unknown classification model: {model_name}")
    else:  # regression
        if model_name == "Linear Regression":
            return LinearRegression(**parameters)
        elif model_name == "Random Forest Regressor":
            return RandomForestRegressor(random_state=42, **parameters)
        elif model_name == "Neural Network":
            return MLPRegressor(
                hidden_layer_sizes=parameters.get("hidden_layer_sizes", (32, 16)),
                max_iter=parameters.get("max_iter", 500),
                activation=parameters.get("activation", "relu"),
                solver=parameters.get("solver", "adam"),
                alpha=parameters.get("alpha", 0.0001),
                learning_rate=parameters.get("learning_rate", "constant"),
                learning_rate_init=parameters.get("learning_rate_init", 0.001),
                random_state=parameters.get("random_state", 42),
            )
        else:
            logger.error(f"Unknown regression model: {model_name}")
            raise ValueError(f"Unknown regression model: {model_name}")


def train_supervised_model(df: pd.DataFrame, target_column: str, task_type: str, model_name: str, scale_features: bool = True, test_size: float = 0.2, model_params: dict = None) -> TrainingResult:
    logger.info(f"Training model: {model_name}")
    logger.info(f"Parameters: {model_params}")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    stratify = y if task_type == "classification" and y.nunique() < 20 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)

    scaler = None
    X_train_fit = X_train.copy()
    X_test_fit = X_test.copy()
    if scale_features:
        scaler = MinMaxScaler() if task_type == "classification" else StandardScaler()
        X_train_fit = scaler.fit_transform(X_train)
        X_test_fit = scaler.transform(X_test)

    model = _get_supervised_model(task_type, model_name, model_params)
    model.fit(X_train_fit, y_train)
    metrics = evaluate_supervised_model(model, X_test_fit, y_test, task_type)

    return TrainingResult(
        model_name=model_name,
        estimator=model,
        task_type=task_type,
        metrics=metrics,
        feature_names=X.columns.tolist(),
        scaler=scaler,
    )

def train_clustering_model(
        df: pd.DataFrame, 
        feature_columns, 
        n_clusters: int, 
        scale_features: bool = True, 
        init: str = "k-means++",
        max_iter: int = 300,
        n_init: int = 10) -> TrainingResult:
    logger.info(f"Training model: {init}")
    X = df[feature_columns].copy()
    scaler = None
    fit_X = X.copy()
    if scale_features:
        scaler = StandardScaler()
        fit_X = scaler.fit_transform(X)
        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            n_init=n_init,
            random_state=42,
        )

    model.fit(fit_X)
    metrics = evaluate_clustering_model(fit_X, model.labels_)
    return TrainingResult(
        model_name=f"KMeans ({n_clusters} clusters)",
        estimator=model,
        task_type="clustering",
        metrics=metrics,
        feature_names=feature_columns,
        scaler=scaler,
    )

