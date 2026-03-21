# Machine Learning Models Hub

## Project Overview
Machine Learning Models Hub is a Streamlit web application showcasing multiple machine learning models across different use cases. It provides an interactive interface for predictions, model comparison, and data visualization.

---

## Features
- Real Estate Price Prediction (Regression)
- Loan Eligibility Prediction (Classification)
- Customer Segmentation (Clustering)
- Neural Network Models (Deep Learning)
- Single & Batch Predictions
- Model Comparison & Metrics
- Interactive Visualizations

---

## Project Structure
```
ml-model-hub/
├── streamlit_app.py           # Main application entry point
├── requirements.txt           # Project dependencies
├── runtime.txt               # Python version specification
├── README.md                 # Project documentation
├── data/                     # Dataset directory
├── notebooks/                # Jupyter notebooks for model development
│   ├── Loan_Eligibility_Model_Solution.ipynb
│   ├── Real_Estate.ipynb
│   ├── UCLA_Neural_Networks_Solution.ipynb
│   └── Unsupervised_Clustering_Solution.ipynb
├── pages/                    # Streamlit multi-page application modules
│   ├── home.py              # Landing page
│   ├── loan_eligibility.py  # Loan prediction interface
│   ├── real_estate.py       # Real estate analysis
│   ├── clustering.py        # Customer segmentation
│   └── neural_networks.py   # Neural network models
├── utils/                    # Utility modules
│   └── model_manager.py     # Model loading and management
└── models/                   # Trained model 
```

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

## Models

### 1. Loan Eligibility
- Task: Classification
- Models: Logistic Regression, Random Forest

### 2. Real Estate
- Task: Regression
- Models: Linear Regression, Random Forest

### 3. Customer Segmentation
- Task: Clustering
- Model: K-Means

### 4. Neural Networks
- Task: Deep Learning
- Model: MLP

---

## Usage

### Single Prediction
1. Select a model page
2. Enter inputs
3. Click predict

### Batch Prediction
1. Upload CSV
2. Select model
3. Download results

---

## Deployment (IMPORTANT)

This app requires:
```
runtime.txt → python-3.13
```
---

## Troubleshooting

### Deployment fails
- Ensure Python version = 3.13
- Ensure dependencies match requirements.txt

### Models not loading
- Run notebooks to regenerate models


## Live Demo

**Try the app:** [Machine Learning Models Hub](https://racheen-ml-model-hubs-streamlit-app-ktl5nz.streamlit.app/)