# Fraud Shield: Credit Card Fraud Detection

![CI](https://github.com/nikhil-kumarrr/Credit-Card-Fraud-Detection/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)

Real-time credit card transaction risk scoring with a machine learning model
and an interactive Streamlit dashboard. The model reaches **0.978 ROC-AUC** on
the public ULB dataset.

![Dashboard](docs/screenshots/dashboard.png)

## Problem

Card fraud is rare but expensive. In this dataset only **492 of 284,807**
transactions (about 0.17%) are fraud, so plain accuracy is misleading: a model
that always predicts "legit" is 99.8% accurate and catches nothing. This project
focuses on recall, precision and ROC-AUC instead.

## Dataset

- Source: Kaggle, "Credit Card Fraud Detection" (ULB Machine Learning Group)
- 284,807 transactions, 492 frauds
- Features: V1 to V28 (anonymised PCA components), Time, Amount
- Target: Class (1 = fraud, 0 = legit)
- The compressed file is kept at data/creditcard.csv.gz because the dashboard reads it

## Approach

1. Scale Amount and Time with StandardScaler
2. Handle the class imbalance with a class-weighted Logistic Regression
3. Evaluate on a stratified hold-out split using ROC-AUC, recall, precision and F1
4. Serve the saved model through a Streamlit dashboard that scores real transactions

## Results

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.978 |

Limitations: the features are anonymised, the data covers only two days of
European card transactions, and a fixed 0.5 threshold is not tuned for cost.
## Project structure

~~~
app/streamlit_app.py        Streamlit dashboard
src/fraud_detection/        data, features, train and predict modules
tests/                      pytest test suite
models/                     saved model, scalers and feature columns
data/                       dataset (see data/README.md)
notebooks/01_eda.ipynb      exploration and model experiments
.github/workflows/ci.yml    lint and tests on every push
~~~

## Installation

~~~
git clone https://github.com/nikhil-kumarrr/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
~~~

On Linux or Mac, activate with `source .venv/bin/activate`.
## Usage

Run the dashboard:

~~~
streamlit run app/streamlit_app.py
~~~

Retrain the model (saved to models/retrained, the deployed model is not overwritten):

~~~
$env:PYTHONPATH="src"
python -m fraud_detection.train
~~~

Score transactions from Python:

~~~python
from fraud_detection.predict import FraudPredictor

predictor = FraudPredictor.from_dir("models")
probabilities = predictor.predict_proba(transactions_df)
~~~

## Development

~~~
pip install -r requirements-dev.txt
pytest
ruff check src tests
~~~

## Future improvements

- Tune the decision threshold for the cost of a missed fraud
- Compare against tree-based models and gradient boosting
- Add precision-recall curves and SHAP explanations to the dashboard

## Author

Nikhil Kumar, MCA, IIT Patna