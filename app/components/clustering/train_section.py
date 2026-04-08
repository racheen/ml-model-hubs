from app.components.common.layout import render_section_title
from app.components.common.training_blocks import render_clustering_training_block
from app.domain.clustering.config import CATEGORY, EXPECTED_FEATURES

def render_train_section():
    render_section_title("Create Your Own Model", "Train a fresh KMeans model on an uploaded dataset.")
    render_clustering_training_block(
        page_key="clustering",
        category=CATEGORY,
        expected_features=EXPECTED_FEATURES,
    )
    
