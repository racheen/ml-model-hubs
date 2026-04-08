from app.components.common.layout import render_section_title
from app.components.common.training_blocks import render_neural_networks_parameters_block, render_supervised_training_block
from app.domain.student_admission.config import CATEGORY, TASK_TYPE, TARGET_COLUMN, DEFAULT_TRAINING_MODELS

def render_train_section():
    render_section_title("Create Your Own Model", "Train a fresh model on an uploaded dataset.")
    render_supervised_training_block(
        page_key="student_admission",
        category=CATEGORY,
        task_type=TASK_TYPE,
        target_column=TARGET_COLUMN,
        training_models=DEFAULT_TRAINING_MODELS,
        csv_file='Admission.csv',
    )
