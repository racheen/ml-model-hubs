CATEGORY = "Student Admission"
TITLE = "Student Admission"
DESCRIPTION = "Predict UCLA admission outcomes with the saved neural network, inspect the model, build your own network, and compare results."
TASK_TYPE = "regression"
USE_JOBLIB = True
DEFAULT_MODELS = ["Neural Network"]
TARGET_COLUMN = "Admit_Chance"
RAW_FORM_FIELDS = {
    "GRE_Score": {"extra": "260-340", "type": "number", "min": 0, "max": 340, "default": 320},
    "TOEFL_Score": {"extra": "0-120", "type": "number", "min": 0, "max": 120, "default": 108},
    "SOP": {"extra": "Statement of Purpose","type": "number", "min": 1.0, "max": 5.0, "default": 4.0, "step": 0.5},
    "LOR": {"extra": "Letter of Recommendation", "type": "number", "min": 1.0, "max": 5.0, "default": 4.0, "step": 0.5},
    "CGPA": {"extra": "0-10","type": "number", "min": 0.0, "max": 10.0, "default": 8.8, "step": 0.1},
    "University_Rating": {"type": "select", "options": [1, 2, 3, 4, 5], "default": 3},
    "Research": {"type": "select", "options": [0, 1], "default": 1},
}
EXPECTED_FEATURES = [
    "GRE_Score","TOEFL_Score","SOP","LOR","CGPA",
    "University_Rating_1","University_Rating_2","University_Rating_3","University_Rating_4","University_Rating_5",
    "Research_0","Research_1",
]
DEFAULT_TRAINING_MODELS = ["Logistic Regression", "Random Forest", "Neural Network"]
