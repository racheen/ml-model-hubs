from app.components.common.module_page import render_module_page
from app.domain.real_estate.config import CATEGORY, TITLE, DESCRIPTION, TASK_TYPE
from .predict_section import render_predict_section
from .analysis_section import render_analysis_section
from .train_section import render_train_section
from .compare_section import render_compare_section

CONFIG = {"CATEGORY": CATEGORY, "TITLE": TITLE, "DESCRIPTION": DESCRIPTION, "TASK_TYPE": TASK_TYPE}

def render_page():
    render_module_page(
        config=CONFIG,
        render_predict=render_predict_section,
        render_analysis=render_analysis_section,
        render_train=render_train_section,
        render_compare=render_compare_section,
    )
