import math
import streamlit as st
import pandas as pd
import joblib

# ================= Page Config =================
st.set_page_config(
    page_title="Fraud Shield",
    page_icon="🛡️",
    layout="wide"
)

# ================= CSS =================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #0a0a0a !important;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header {visibility: hidden;}

* { color: #f3f4f6 !important; }

.block-container {
    max-width: 980px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    margin: 0 auto !important;
}

@keyframes riseUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
.hero-wrap { text-align: center; padding: 6px 0 20px 0; }
.hero-badge {
    display: inline-block;
    background: rgba(234, 179, 8, 0.08);
    border: 1px solid rgba(234, 179, 8, 0.35);
    color: #eab308 !important;
    font-size: 12.5px;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 999px;
    margin-bottom: 16px;
    animation: riseUp 0.6s ease-out;
}
.hero-badge .dot { display: inline-block; width: 7px; height: 7px; background: #eab308; border-radius: 50%; margin-right: 8px; }
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    font-weight: 800;
    line-height: 1.15;
    margin: 0;
    animation: riseUp 0.7s ease-out 0.1s both;
}
.hero-title .accent { color: #eab308 !important; }
.hero-sub { color: #9ca3af !important; font-size: 15px; margin-top: 10px; animation: riseUp 0.7s ease-out 0.25s both; }
.hero-pills { display: flex; justify-content: center; gap: 10px; margin-top: 18px; flex-wrap: wrap; animation: riseUp 0.7s ease-out 0.4s both; }
.hero-pill { background: #171717; border: 1px solid #2e2e2e; border-radius: 9px; padding: 8px 14px; font-size: 12.5px; font-weight: 600; color: #d1d5db !important; }
.hero-pill.gold { background: #eab308 !important; color: #0a0a0a !important; border: none; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #171717 !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 14px !important;
    padding: 6px 6px !important;
    margin-bottom: 14px !important;
}
.input-card-title { font-size: 12.5px; font-weight: 700; color: #9ca3af !important; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px; }

div[data-testid="stSlider"] > div > div > div > div { background: linear-gradient(90deg, #eab308, #f59e0b) !important; }
div[data-testid="stSlider"] [role="slider"] {
    background: #eab308 !important;
    border: 3px solid #0a0a0a !important;
    box-shadow: 0 0 0 3px rgba(234, 179, 8, 0.35), 0 0 12px rgba(234, 179, 8, 0.5) !important;
    width: 20px !important;
    height: 20px !important;
}
div[data-testid="stSlider"] > div > div > div { background: #2e2e2e !important; }

.stButton button { background: #eab308 !important; border-radius: 10px; padding: 10px 20px; border: none; width: 100%; font-weight: 700; }
.stButton button p, .stButton button span, .stButton button div { color: #0a0a0a !important; }
.stButton button:hover { background: #fbbf24 !important; }

.result-card { background: #171717 !important; border-radius: 14px; padding: 18px 22px; border-left: 6px solid #10b981; box-shadow: 0 1px 3px rgba(0,0,0,0.4); margin-bottom: 12px; }
.result-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #9ca3af !important; letter-spacing: 1.2px; margin-bottom: 5px; }
.result-classification { font-size: 22px; font-weight: 800; margin-bottom: 8px; }
.result-desc { font-size: 13px; color: #d1d5db !important; line-height: 1.45; }

.stat-value { font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 700; }
.stat-label { font-size: 11px; font-weight: 600; color: #9ca3af !important; text-transform: uppercase; letter-spacing: 0.7px; margin-top: 2px; }

@keyframes ringFill { from { stroke-dashoffset: var(--circumference); } to { stroke-dashoffset: var(--offset); } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
.ring-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 4px 0; }
.ring-svg { transform: rotate(-90deg); }
.ring-track { fill: none; stroke: #2e2e2e; stroke-width: 12; }
.ring-progress { fill: none; stroke-width: 12; stroke-linecap: round; animation: ringFill 1.1s ease-out forwards; }
.ring-center-text { position: relative; margin-top: -96px; text-align: center; animation: fadeIn 1s ease-out 0.4s both; }
.ring-pct { font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 700; }
.ring-caption { font-size: 11px; color: #9ca3af !important; margin-top: 1px; }
.gauge-tag { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:10.5px; font-weight:700; color:#9ca3af !important; background:#171717 !important; border:1px solid #2e2e2e; border-radius:999px; padding:3px 11px; margin-bottom:8px; }

.factors-title { font-size: 15px; font-weight: 800; margin: 16px 0 10px 0; }
.factors-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.factor-row { display: flex; align-items: flex-start; gap: 10px; background: #171717 !important; border: 1px solid #2e2e2e; border-radius: 12px; padding: 12px 14px; }
.factor-icon { font-size: 16px; line-height: 1; margin-top: 1px; }
.factor-text-title { font-size: 12.5px; font-weight: 700; margin-bottom: 2px; }
.factor-text-sub { font-size: 11.5px; color: #9ca3af !important; }
.factor-push-up { color: #f87171 !important; font-weight: 700; }
.factor-push-down { color: #4ade80 !important; font-weight: 700; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ================= Load Model Artifacts =================
model = joblib.load('fraud_detection_model.pkl')
scaler_amount = joblib.load('scaler_amount.pkl')
scaler_time = joblib.load('scaler_time.pkl')
feature_columns = joblib.load('feature_columns.pkl')


@st.cache_data
def load_data():
    return pd.read_csv('creditcard.csv.gz')


@st.cache_data
def compute_reference_stats(_df):
    v_cols = [c for c in feature_columns if c.startswith('V')]
    v_mean = _df[v_cols].mean()
    fraud_mean = _df[_df['Class'] == 1][v_cols].mean()
    avg_amount = _df['Amount'].mean()
    return v_cols, v_mean, fraud_mean, avg_amount


df = load_data()
v_columns, v_mean, fraud_mean, avg_amount = compute_reference_stats(df)


def get_scaled_row(raw_row: pd.Series, amount_override: float = None, hour_override: int = None):
    """
    Build a model-ready input row from a REAL transaction pulled from the dataset.
    All V1-V28 values are the genuine PCA features from that row - nothing is
    interpolated or reverse-engineered to hit a target score.
    Only Amount/Time can optionally be overridden by the user for what-if analysis.
    """
    row = raw_row[feature_columns].copy()
    if amount_override is not None:
        row['Amount'] = amount_override
    if hour_override is not None:
        row['Time'] = float(hour_override * 3600)

    input_df = pd.DataFrame([row])[feature_columns]
    input_df['Amount'] = scaler_amount.transform(input_df[['Amount']])
    input_df['Time'] = scaler_time.transform(input_df[['Time']])
    return input_df


def sample_transaction(kind: str):
    """Pull one real, random transaction from the dataset. kind = 'random' | 'legit' | 'fraud'."""
    if kind == 'fraud':
        pool = df[df['Class'] == 1]
    elif kind == 'legit':
        pool = df[df['Class'] == 0]
    else:
        pool = df
    return pool.sample(n=1).iloc[0]


# ================= Hero Section =================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge"><span class="dot"></span>Credit Card Fraud Detection</div>
    <h1 class="hero-title">Fraud <span class="accent">Shield</span></h1>
    <div class="hero-sub">Real-time transaction risk detection, powered by Machine Learning</div>
    <div class="hero-pills">
        <div class="hero-pill gold">Bank-Grade Model</div>
        <div class="hero-pill">97.8% AUC Score</div>
        <div class="hero-pill">Instant Results</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= Inputs =================
if 'txn' not in st.session_state:
    st.session_state.txn = sample_transaction('random')

with st.container(border=True):
    st.markdown('<div class="input-card-title">Pick a Real Transaction</div>', unsafe_allow_html=True)
    st.caption(
        "This pulls an actual row (with its real V1-V28 PCA features) from the dataset, "
        "so the model genuinely scores real transaction data - nothing is faked or reverse-engineered."
    )
    b1, b2, b3 = st.columns(3)
    if b1.button("Random Transaction"):
        st.session_state.txn = sample_transaction('random')
    if b2.button("Sample a Legit One"):
        st.session_state.txn = sample_transaction('legit')
    if b3.button("Sample a Known Fraud"):
        st.session_state.txn = sample_transaction('fraud')

txn = st.session_state.txn
actual_hour = int((txn['Time'] % 86400) // 3600)

with st.container(border=True):
    st.markdown('<div class="input-card-title">Transaction Details (editable)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("Amount ($)", min_value=0.0, value=float(txn['Amount']), step=1.0)
    with c2:
        hour_of_day = st.slider("Hour of Day", 0, 23, actual_hour)
    st.caption(
        f"Ground truth for this sampled row: **{'FRAUD' if txn['Class'] == 1 else 'LEGIT'}** "
        "(hidden from the model - shown here only so you can verify predictions against reality)."
    )

analyze = st.button("Analyze Transaction")


# ================= Circular Ring Builder =================
def build_ring(pct, size=170, stroke=12):
    pct = max(0, min(100, pct))
    r = (size - stroke) / 2
    circumference = 2 * math.pi * r
    offset = circumference * (1 - pct / 100)
    cx = cy = size / 2

    if pct < 30:
        color = "#4ade80"
    elif pct < 70:
        color = "#facc15"
    else:
        color = "#f87171"

    html = f"""
    <div class="ring-wrap">
        <svg class="ring-svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
            <circle class="ring-track" cx="{cx}" cy="{cy}" r="{r}"></circle>
            <circle class="ring-progress" cx="{cx}" cy="{cy}" r="{r}"
                stroke="{color}"
                stroke-dasharray="{circumference}"
                style="--circumference:{circumference}; --offset:{offset}; stroke-dashoffset:{offset};">
            </circle>
        </svg>
        <div class="ring-center-text">
            <div class="ring-pct" style="color:{color} !important;">{pct:.1f}%</div>
            <div class="ring-caption">Fraud Probability</div>
        </div>
    </div>
    """
    return html


# ================= Result Section =================
if analyze:
    input_df = get_scaled_row(txn, amount_override=amount, hour_override=hour_of_day)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    pct = probability * 100

    if pct < 30:
        classification = "LOW RISK"
        border_color = "#10b981"
        desc = "this transaction is <b>likely legitimate</b> based on the submitted profile."
    elif pct < 70:
        classification = "MEDIUM RISK"
        border_color = "#f59e0b"
        desc = "this transaction shows <b>some unusual signals</b> and may warrant a manual review."
    else:
        classification = "HIGH RISK"
        border_color = "#ef4444"
        desc = "this transaction shows <b>strong indicators of fraud</b> and should be flagged immediately."

    left, right = st.columns([1.1, 1])

    with left:
        result_html = (
            f'<div class="result-card" style="border-left-color:{border_color};">'
            f'<div class="result-eyebrow">RISK CLASSIFICATION</div>'
            f'<div class="result-classification">{classification}</div>'
            f'<div class="result-desc">Based on the submitted transaction, {desc}</div>'
            f'</div>'
        )
        st.markdown(result_html, unsafe_allow_html=True)

        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                f'<div class="stat-value">{probability*100:.1f}%</div>'
                f'<div class="stat-label">Fraud Probability</div>',
                unsafe_allow_html=True
            )
        with s2:
            st.markdown(
                f'<div class="stat-value">{(1-probability)*100:.1f}%</div>'
                f'<div class="stat-label">Legit Probability</div>',
                unsafe_allow_html=True
            )

    with right:
        st.markdown('<div style="text-align:center;"><div class="gauge-tag">🛡️ RISK SCORE</div></div>', unsafe_allow_html=True)
        st.markdown(build_ring(pct), unsafe_allow_html=True)

    st.markdown('<div class="factors-title">Key Contributing Factors</div>', unsafe_allow_html=True)

    coefs = model.coef_[0]
    input_values = input_df.iloc[0].values
    contributions = pd.Series(coefs * input_values, index=feature_columns)
    top_contribs = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(4)

    friendly_names = {
        'Amount': f"Transaction Amount (${amount:,.2f})",
        'Time': f"Transaction Hour ({hour_of_day}:00)"
    }

    factors_html = '<div class="factors-grid">'
    for feat, val in top_contribs.items():
        direction_up = val > 0
        icon = "⚠️" if direction_up else ""
        arrow_class = "factor-push-up" if direction_up else "factor-push-down"
        arrow_text = "increases risk" if direction_up else "decreases risk"
        label = friendly_names.get(feat, f"Internal Risk Signal ({feat})")
        factors_html += (
            f'<div class="factor-row">'
            f'<div class="factor-icon">{icon}</div>'
            f'<div><div class="factor-text-title">{label}</div>'
            f'<div class="factor-text-sub">This factor <span class="{arrow_class}">{arrow_text}</span> '
            f'(score: {abs(val):.2f})</div></div>'
            f'</div>'
        )
    factors_html += '</div>'
    st.markdown(factors_html, unsafe_allow_html=True)

# ================= Footer =================
st.markdown("<br><hr style='border-color:#2e2e2e;'>", unsafe_allow_html=True)
f1, f2, f3 = st.columns(3)
f1.metric("Total Transactions", f"{len(df):,}")
f2.metric("Fraud Cases", f"{int(df['Class'].sum())}")
f3.metric("Model AUC", "0.978")