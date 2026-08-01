<div align="center">

# 🧠 AttritionSense
### Predicting Employee Attrition Before It Happens

An end-to-end machine learning system that flags at-risk employees using HR data — turning reactive HR into proactive retention strategy.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

[**Live Demo**](https://ibm-attrition-predictor.streamlit.app/) · [**Report a Bug**](#) · [**Notebook**](./IBM_Attrition.ipynb)

</div>

---

## Why This Matters

Replacing an employee typically costs **50–200% of their annual salary** once you factor in recruiting, onboarding, and lost productivity. Most companies only find out someone is a flight risk *after* they've already resigned.

**AttritionSense** flips that timeline — it scores every employee's attrition risk in real time from HR data that companies already collect, so HR teams can intervene weeks or months before someone walks out the door.

<div align="center">
<img src="https://github.com/nikhil-kumarrr/images/blob/main/Screenshot%202026-07-02%20132132.png?raw=true" width="800"/>
</div>

---

## What It Does

- Takes an employee's profile (role, income, tenure, satisfaction scores, overtime status, etc.)
- Runs it through a trained Logistic Regression model
- Returns a **calibrated attrition probability**, a risk tier (Low / Medium / High), and a **suggested HR action plan**
- Built as an interactive Streamlit dashboard — no coding required to use it

<div align="center">
<img src="https://github.com/nikhil-kumarrr/images/blob/main/Screenshot%202026-07-02%20132207.png?raw=true" width="800"/>
</div>

---

## Results

Four models were trained and compared using 5-fold stratified cross-validation on the IBM HR Analytics dataset (1,470 employees):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Logistic Regression** | **86.4%** | 62.1% | 38.3% | **47.4%** | **82.9%** |
| Gradient Boosting | 84.7% | 52.9% | 38.3% | 44.4% | 78.4% |
| Random Forest | 83.7% | 48.2% | 27.7% | 35.1% | 77.3% |
| Decision Tree | 76.9% | 31.6% | 38.3% | 34.6% | 71.4% |

**Logistic Regression won** — not because it's the most powerful algorithm, but because it had the best F1/ROC-AUC *and* gives HR stakeholders interpretable coefficients ("overtime increases risk by X") instead of a black box. In an HR context, explainability is often as important as raw accuracy.

> **Honest caveat:** Recall (38.3%) means the model still misses roughly 6 in 10 employees who actually leave. This is a known challenge with attrition data — the signal is inherently noisy because people leave for reasons no dataset captures (a competing offer, a personal move, a bad manager interaction last week). SMOTE was used to address class imbalance during training; further gains would likely need richer features (engagement survey text, manager 1:1 notes, market salary benchmarks) rather than a different algorithm.

---

## Key Findings

- **Overtime is the single strongest driver** — employees working overtime leave at a substantially higher rate
- **Job satisfaction and work-life balance scores** are strong early-warning signals, often more predictive than income
- **Pay relative to role/experience** matters more than absolute salary — being underpaid *for your level* drives attrition
- **Tenure is protective** — attrition risk drops sharply after the first couple of years
- **Sales Representatives** show a notably higher attrition rate than any other role, worth a targeted retention strategy

---

## How It's Built

**Pipeline:**
```
Raw HR data → Cleaning & feature engineering → Encoding & scaling
→ SMOTE (train set only) → Train 4 models w/ 5-fold CV
→ Select best model → Serialize (model/scaler/columns) → Serve via Streamlit
```

**Feature engineering highlights:**
- Dropped constant/leak-prone columns (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`)
- Engineered features: `TenureRatio`, `PromotionGap`, `SatisfactionScore`, `IncomePerYear`
- Class imbalance handled with **SMOTE — applied only to the training fold**, never the test set, to avoid inflating evaluation metrics with synthetic leakage

| Layer | Tool |
|---|---|
| Language | Python |
| Data wrangling | Pandas, NumPy |
| EDA | Matplotlib, Seaborn |
| Modeling | scikit-learn |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Serialization | Joblib |
| Dashboard | Streamlit |

---

## Project Structure

```
attrition-predictor/
│
├── app.py                                   ← Streamlit dashboard
├── IBM_Attrition.ipynb                      ← Full ML pipeline notebook
├── WA_Fn-UseC_-HR-Employee-Attrition.csv     ← Raw dataset
│
├── model.pkl                                ← Trained Logistic Regression model
├── scaler.pkl                               ← StandardScaler
├── columns.pkl                              ← Feature name/order list
│
├── requirements.txt
└── README.md
```

---

## Run It Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/IBM-hr-attrition-predictor.git
cd IBM-hr-attrition-predictor

# 2. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (First time only) run the notebook to generate model artifacts
jupyter notebook IBM_Attrition.ipynb

# 5. Launch the dashboard
streamlit run app.py
```

---

## Dataset

[IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — 1,470 employee records across Sales, R&D, and HR departments, publicly available on Kaggle.

> Independent project built on IBM's publicly available dataset. Not affiliated with or endorsed by IBM.

---

## What I'd Improve With More Time

- Add SHAP-based per-prediction explanations instead of only global coefficients
- Try threshold tuning to better balance precision/recall for HR's actual use case (would they rather over-flag or under-flag?)
- Incorporate a time dimension — attrition is a *when*, not just a *whether*, and survival analysis could model that
- Add authentication and a proper database backend for real HR-tool deployment

---

## License

MIT — see [LICENSE](./LICENSE) for details.

<div align="center">

Built by [Your Name](https://github.com/your-username) · [LinkedIn](#) · [Portfolio](#)

</div>
