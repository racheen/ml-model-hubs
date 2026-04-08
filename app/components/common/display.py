from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st
from pathlib import Path
from app.config.paths import MODELS_DIR
from app.domain.clustering.config import DEFAULT_MODELS

def show_metric_cards(metrics: dict):
    if not metrics:
        st.info("No metrics available yet.")
        return
    cols = st.columns(len(metrics))
    for idx, (label, value) in enumerate(metrics.items()):
        cols[idx].metric(label.upper(), value)

def show_dataframe(df: pd.DataFrame, height: int = 280):
    if df is None or df.empty:
        st.info("Nothing to display.")
        return
    st.dataframe(df, use_container_width=True, height=height)

def show_probability(probabilities, label: str = "Approval Probability"):
    if not probabilities:
        return
    st.metric(label, f"{probabilities*100:.2f}%")


def plot_clusters(df, model, user_point=None, feature_x="Annual_Income", feature_y="Spending_Score", scaler=None):
     # Get all features that the model expects
    if hasattr(model, 'feature_names_in_'):
        required_features = model.feature_names_in_
    else:
        # Fallback to common features
        required_features = ["Age", "Annual_Income", "Spending_Score"]
    
    # Prepare data with all required features
    X_full = df[required_features].copy()
    
    # Apply scaling if scaler was used during training
    if scaler is not None:
        X_scaled = scaler.transform(X_full)
        clusters = model.predict(X_scaled)
    else:
        clusters = model.predict(X_full)
    
    # Extract the two features for plotting (from original data)
    X_plot = df[[feature_x, feature_y]]
    
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot of all points
    scatter = ax.scatter(
        X_plot[feature_x],
        X_plot[feature_y],
        c=clusters,
        cmap='viridis',
        alpha=0.6,
        s=60,
        edgecolors='w',
        linewidth=0.5
    )

    # Get cluster centers
    centers = model.cluster_centers_
    
    # Find indices of the features we're plotting
    feature_indices = [list(required_features).index(feature_x), 
                      list(required_features).index(feature_y)]
    
    # Extract only the relevant dimensions from centers
    centers_2d = centers[:, feature_indices]
    
    # If scaler was used, inverse transform the centers for plotting
    if scaler is not None:
        # Create a full feature array with centers
        centers_full = centers.copy()
        centers_full_original = scaler.inverse_transform(centers_full)
        centers_2d = centers_full_original[:, feature_indices]

    # Plot cluster centers
    ax.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        marker="X",
        s=300,
        c='red',
        edgecolors='black',
        linewidths=2,
        label="Cluster Centers",
        zorder=10
    )

    # Annotate cluster centers
    for i, (x, y) in enumerate(centers_2d):
        ax.annotate(
            f'C{i}',
            (x, y),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color='white',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='red', edgecolor='black', linewidth=2)
        )

    # Plot user point if provided
    if user_point is not None:
        ax.scatter(
            user_point[0],
            user_point[1],
            marker="*",
            s=500,
            c='gold',
            edgecolors='black',
            linewidths=2,
            label="Your Input",
            zorder=11
        )
        
        # Add annotation for user point
        ax.annotate(
            'YOU',
            (user_point[0], user_point[1]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2)
        )

    ax.set_xlabel(feature_x.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel(feature_y.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title("Customer Segmentation Clusters", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster', rotation=270, labelpad=15)
    
    fig.tight_layout()

    return fig

def get_available_models(category: str, use_joblib: bool = True) -> list:
    """Get list of available model files for a category."""
    category_dir = Path(MODELS_DIR) / category
    if not category_dir.exists():
        return DEFAULT_MODELS
    
    extension = ".pkl"
    model_files = list(category_dir.glob(f"*{extension}"))
    
    if not model_files:
        return DEFAULT_MODELS
    
    model_names = []
    for f in model_files:
        # Skip scaler files
        if f.stem.lower() == "scaler" or "scaler" in f.stem.lower():
            continue
        
        # Remove '_model' suffix if present
        name = f.stem.replace("_model", "")
        model_names.append(name)
    return sorted(model_names)