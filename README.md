# Machine Learning Models Hub

## Project Overview
ML Models Hub is a modular Streamlit application that brings together four machine learning workflows in one interface:

- Loan Eligibility
- Real Estate
- Student Admission
- Customer Segmentation

The project adapts notebook-based machine learning solutions into a structured application with reusable components, centralized model handling, and a consistent module design for prediction, analysis, tuning, and comparison.

**Try the app:** [Machine Learning Models Hub](https://racheen-ml-model-hubs-streamlit-app-ninwjb.streamlit.app/)

---

## Features

- Single-entry Streamlit application
- Manual sidebar navigation using radio buttons
- Four machine learning modules
- Shared module layout across workflows
- Default model prediction
- Model analysis views
- Hyperparameter tuning using built-in datasets
- Default vs tuned model comparison
- Centralized model and artifact loading
- Clean layered architecture

---

## Project Structure
```bash
ml-model-hubs/
│
├── streamlit_app.py              # Main application entry point
├── README.md
├── requirements.txt
├── runtime.txt
│
├── app/
│   ├── components/
│   │   ├── common/ 
│   │   │   ├── analysis_helpers.py
│   │   │   ├── display.py
│   │   │   ├── forms.py
│   │   │   ├── home.py
│   │   │   ├── layout.py
│   │   │   ├── module_page.py
│   │   │   └── training_blocks.py
│   │   ├── loan_eligibility/
│   │   │   ├── analysis_section.py
│   │   │   ├── compare_section.py
│   │   │   ├── page.py
│   │   │   ├── predict_section.py
│   │   │   └── train_section.py
│   │   ├── real_estate/
│   │   ├── student_admission/
│   │   └── customer_segmentation/
│   │
│   ├── config/
│   │   ├── paths.py
│   │   └── settings.py
│   ├── core/
│   │   ├── schemas.py
│   │   └── state_manager.py
│   ├── domain/
│   │   ├── loan_eligibility/
│   │   │   ├── config.py
│   │   │   └── preprocess.py
│   │   ├── real_estate/
│   │   ├── student_admission/
│   │   └── customer_segmentation/
│   ├── repositories/
│   │   ├── legacy_model_manager.py
│   │   └── model_repository.py
│   ├── services/
│   │   ├── comparison_service.py
│   │   ├── evaluation_service.py
│   │   ├── prediction_service.py
│   │   └── training_service.py
│   └── utils/
│       └── formatter.py
├── data/                         # Built-in datasets used for training/
├── models/                       # Serialized trained models and scalers
├── notebooks/                    # Original notebook workflows
├── reports/
└── report.pdf
```

---

## Architecture

The system follows a layered design:

```text
User
  ↓
streamlit_app.py
  ↓
UI Components
  ↓
Services
  ↓
Repositories
  ↓
Models / Data / Artifacts
```

### Layer Responsibilities

#### `streamlit_app.py`
- configures the app
- renders sidebar navigation
- routes the selected module

#### `app/components/`
- handles Streamlit UI rendering
- keeps page/module code readable and modular

#### `app/services/`
- contains prediction, training, evaluation, and comparison workflows

#### `app/repositories/`
- loads models, scalers, and related metadata from disk

#### `app/domain/`
- stores problem-specific settings, preprocessing logic, and model configuration

---

## Module Workflow

Each module follows the same user-facing structure:

### 1. Predict
Run inference using a default trained model.

### 2. Model Analysis
Review task-relevant metrics, model details, or visualization outputs.

### 3. Tune / Custom Model
Adjust parameters and retrain using the built-in dataset for that module.

### 4. Compare
Compare the default model against the tuned/custom model.

---

## Modules

| Module | Task | Default Models |
|---|---|---|
| Loan Eligibility | Classification | Logistic Regression, Random Forest |
| Real Estate | Regression | Linear Regression, Random Forest |
| Student Admission | Neural Network Prediction | MLP / Neural Network |
| Customer Segmentation | Clustering | K-Means |

---

## Installation

### Prerequisites
- Python 3.13
- pip

### Setup
```bash
git clone https://github.com/yourusername/ml-model-hubs.git
cd ml-model-hubs

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Dependencies
- streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib
- seaborn
- pillow

---

## Tech Stack

- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Matplotlib
- Jupyter Notebooks
