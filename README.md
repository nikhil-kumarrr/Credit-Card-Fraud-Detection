# Fraud Shield: Credit Card Fraud Detection

![CI](https://github.com/nikhil-kumarrr/Credit-Card-Fraud-Detection/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)

Real-time credit card transaction risk scoring with a machine learning model
and an interactive Streamlit dashboard. The model reaches **0.978 ROC-AUC** on
the public ULB dataset.

**Live demo:** PASTE_YOUR_STREAMLIT_LINK_HERE

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

1. Random undersampling: keep all 492 frauds and 492 random legit transactions (984 rows)
2. Scale Amount and Time with StandardScaler
3. Stratified 80/20 split, then compare Logistic Regression, Decision Tree, Random Forest and KNN
4. Choose Logistic Regression: it matches Random Forest on test accuracy without overfitting (train 96.1% vs 100% for the tree models)
5. Re-evaluate the chosen model on the real, imbalanced data (every row not used for training)
6. Serve the saved model through a Streamlit dashboard that scores real transactions

## Results

Model comparison on the balanced test set (197 transactions):

| Model | Train accuracy | Test accuracy |
| --- | --- | --- |
| Logistic Regression | 96.1% | 93.4% |
| Decision Tree | 100% | 89.3% |
| Random Forest | 100% | 93.4% |
| KNN | 95.3% | 92.4% |

Logistic Regression reaches **0.978 ROC-AUC** on the balanced test set (fraud precision 0.97, recall 0.90).

The balanced test set hides the real difficulty, so the model is also checked on the original imbalanced data (284,020 transactions, 98 frauds):

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.977 |
| Fraud recall | 0.898 |
| Fraud precision | 0.0096 |
| PR-AUC | 0.354 |

At the default 0.5 threshold the model catches about 90% of frauds, but flags roughly 100 legit transactions for every real fraud. Tuning the threshold for the cost of a false alarm is the main next step.

Limitations: the features are anonymised and the data covers only two days of European card transactions.

## Project structure

~~~
app/streamlit_app.py        Streamlit dashboard
src/fraud_detection/        data, features, train, predict and FastAPI modules
tests/                      pytest test suite
models/                     saved model, scalers and feature columns
data/                       dataset (see data/README.md)
notebooks/01_eda.ipynb      exploration and model experiments
.github/workflows/ci.yml    lint, format check and tests on every push
CONTRIBUTING.md             how to set up, test and contribute
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

## API

A small FastAPI service serves the same model over HTTP.

~~~
pip install -r requirements-dev.txt
uvicorn fraud_detection.api:app --app-dir src
~~~

Interactive docs are available at http://127.0.0.1:8000/docs.

| Endpoint | Purpose |
| --- | --- |
| GET /health | Service status and version |
| POST /predict?threshold=0.5 | Fraud probability, decision and risk level for 1 to 1000 transactions |

Example request (a transaction needs Time, Amount and V1 to V28):

~~~python
import requests

row = {"Time": 0.0, "Amount": 149.62, **{f"V{i}": 0.0 for i in range(1, 29)}}
response = requests.post("http://127.0.0.1:8000/predict", json={"transactions": [row]})
print(response.json())
# {"threshold": 0.5, "predictions": [{"fraud_probability": ..., "is_fraud": ..., "risk_level": "low"}]}
~~~

Invalid input (negative amount, missing feature, empty batch, threshold outside 0 to 1) is rejected with HTTP 422.

## Development

~~~
pip install -r requirements-dev.txt
pytest --cov
ruff check src tests
ruff format src tests
~~~

The same lint, format and test steps run on every push through GitHub Actions.

## Model card

- Intended use: an educational and portfolio demo of fraud scoring. It is not meant for real payment decisions.
- Training data: the public ULB credit card dataset (European cardholders, September 2013, two days of transactions). Features V1 to V28 are anonymised PCA components, so there are no personal details.
- Training set: 984 rows (492 frauds and 492 random legit transactions) after random undersampling.
- Performance: see Results above. ROC-AUC is about 0.977 on the real imbalanced data, but precision at the 0.5 threshold is under 1%.
- Known limitations: undersampling throws away most legit transactions, the data is old and covers a short period, and the dashboard explanations only show model coefficients on anonymised features.

## Future improvements

- Tune the decision threshold for the cost of a missed fraud versus a false alarm
- Try gradient boosting and SMOTE or cost-sensitive learning instead of undersampling
- Add precision-recall curves and SHAP explanations to the dashboard
- Add a Dockerfile for the API

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Author

Nikhil Kumar, MCA, IIT Patna
