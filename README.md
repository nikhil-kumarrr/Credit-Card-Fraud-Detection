# Credit Card Fraud Detection  
  
A ML-powered credit card fraud detection system built for real-time transaction risk scoring using Logistic Regression, Random Forest, Decision Tree, and KNN — with an interactive **Fraud Shield** dashboard built in Streamlit.

Pick a real transaction from the dataset and instantly get an AI-powered fraud risk prediction with probability scores and the key factors driving the decision.

---

## Features

- ML-based real-time transaction fraud risk prediction
- 4 models trained and compared (Logistic Regression · Decision Tree · Random Forest · KNN)
- Instant risk classification with probability scores
- Fraud probability gauge, risk factor breakdown
- Real transaction sampling (random / known legit / known fraud) — no fake or simulated inputs
- Professional dark UI dashboard
- Uses real-world Kaggle Credit Card Fraud dataset (284,807 transactions)
- Real-time prediction engine with saved model artifacts

---

## How It Works

### 1️⃣ Dataset

**creditcard.csv** — 284,807 European cardholder transactions (Sept 2013), features anonymized via PCA

| Feature | Description |
|---|---|
| Time | Seconds elapsed since first transaction in dataset |
| Amount | Transaction amount ($) |
| V1–V28 | Anonymized PCA-transformed features |
| Class | **Target** — Fraud (1) / Legit (0) |

---

### 2️⃣ Data Processing (Notebook)

- Checked for missing values and duplicates
- Feature scaling → `StandardScaler` applied separately to `Amount` and `Time`
- Class imbalance handling → Undersampling (492 fraud vs. 492 randomly sampled legit transactions)
- Train-test split → 80/20, stratified by target
- **Real-world evaluation** → Same trained model re-evaluated on the full imbalanced holdout set (excluding training rows) to measure true production-level performance

---

### 3️⃣ EDA Performed

- Class distribution (Fraud vs Legit) — highlighting extreme imbalance (~0.17%)
- Transaction Amount distribution by class
- Transaction Time patterns for fraud vs legit
- Correlation heatmap across PCA features

---

### 4️⃣ ML Models

- **4 Models Trained** → Logistic Regression, Decision Tree, Random Forest, KNN
- **Evaluation** → Confusion Matrix, Precision-Recall, ROC-AUC on balanced test set, followed by real-world imbalanced holdout evaluation
- **Best Model** → Logistic Regression (Accuracy: 93.4% | ROC-AUC: 0.978, lowest overfitting)
- **Saved as** → `fraud_detection_model.pkl` + `scaler_amount.pkl` + `scaler_time.pkl` + `feature_columns.pkl`

---

## Model Results

| Model | Accuracy | ROC-AUC | Notes |
|---|---|---|---|
| **Logistic Regression** | **93.4%** | **0.978** | Lowest overfitting, most reliable |
| Random Forest | ~93% | 0.979 | Comparable AUC, signs of overfitting |
| Decision Tree | Lower | Lower | Overfit the most |
| KNN | Lower | Lower | Least consistent |

> Logistic Regression selected as best model — lowest overfitting among all models tested, with directly interpretable coefficients.

**Real-world holdout check** (full imbalanced dataset, 98 fraud cases out of 284,020 transactions):

| Metric | Score |
|---|---|
| Recall (Fraud) | 89.8% |
| Precision (Fraud) | 0.96% |
| ROC-AUC | 0.977 |
| PR-AUC | 0.354 |

> Balanced-test metrics look strong, but they don't reflect production. On the true imbalanced distribution, the model catches most fraud (high recall) but flags many false positives (low precision) — a known trade-off of undersampling on extreme class imbalance.

---

## Key Findings

- A handful of anonymized PCA features (V-columns) drive most of the model's decisions, far more than raw Amount or Time
- Undersampling enables fast, effective training but must be paired with imbalanced-data evaluation, or reported metrics silently overstate real-world performance
- The model generalizes well by ROC-AUC (0.977) even though precision at the default threshold is low — this reflects the extreme class imbalance, not poor model quality
- Threshold tuning or cost-sensitive learning would be the natural next step to improve precision without sacrificing recall

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | EDA visualizations |
| Scikit-learn | ML models, preprocessing, evaluation |
| Joblib | Model serialization |
| Streamlit | Interactive web dashboard |

---

## Project Structure
```
Credit Card Fraud Detection/
│
├── app.py ← Streamlit dashboard
├── Credit_Card_Fraud_Detection.ipynb ← Full ML pipeline + real-world evaluation
├── creditcard.csv.gz ← Compressed dataset (284K+ transactions)
│
├── fraud_detection_model.pkl ← Trained Logistic Regression model
├── scaler_amount.pkl ← StandardScaler for Amount
├── scaler_time.pkl ← StandardScaler for Time
├── feature_columns.pkl ← Feature name/order list
│
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/fraudshield-ml.git
cd fraudshield-ml
```

### 2️⃣ Create virtual environment
```bash
python -m venv venv
```

### 3️⃣ Activate environment

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 4️⃣ Install requirements
```bash
pip install -r requirements.txt
```

### 5️⃣ Run the notebook first (to reproduce model files)
```bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

### 6️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---

## requirements.txt
```
streamlit==1.38.0
pandas==2.2.2
scikit-learn==1.5.1
joblib==1.4.2
numpy==1.26.4
```

---

## Dataset

Available on Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> Independent project built using a publicly available Kaggle dataset. Not affiliated with or endorsed by the dataset's original publishers.
---

## Screenshots
![img alt](https://github.com/nikhil-kumarrr/images/blob/main/Screenshot%202026-08-01%20151319.png?raw=true)
![img alt](https://github.com/nikhil-kumarrr/images/blob/main/Screenshot%202026-08-01%20151345.png?raw=true)
