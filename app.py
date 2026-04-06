"""
Loan Default Prediction — Streamlit App
Standalone application: no local imports required.
"""
import streamlit as st
import pandas as pd

import joblib
import os

MODEL_DIR = "models/"


# ── Load best model ──
@st.cache_resource

def load_model():
    best_model_path = os.path.join(MODEL_DIR, "best_model.txt")
    with open(best_model_path, "r") as f:
        best_name = f.read().strip()
    model = joblib.load(os.path.join(MODEL_DIR, f"{best_name}.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    return model, scaler, best_name


model, scaler, active_model = load_model()

# ── Page config ──
st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Loan Default Risk Predictor")
st.markdown("Enter customer details below to estimate the probability of loan default.")

# ── Sidebar inputs ──
st.sidebar.header("Customer Features")

credit_lines = st.sidebar.slider(
    "Credit Lines Outstanding", min_value=0, max_value=10, value=3, step=1
)
loan_amt = st.sidebar.slider(
    "Loan Amount Outstanding ($)", min_value=0, max_value=50000, value=15000, step=500
)
total_debt = st.sidebar.slider(
    "Total Debt Outstanding ($)", min_value=0, max_value=100000, value=30000, step=1000
)
income = st.sidebar.slider(
    "Annual Income ($)", min_value=10000, max_value=200000, value=65000, step=1000
)
years_employed = st.sidebar.slider(
    "Years Employed", min_value=0, max_value=40, value=8, step=1
)
fico_score = st.sidebar.slider(
    "FICO Score", min_value=300, max_value=850, value=680, step=5
)

# ── Predict ──
if st.button("🔍 Predict Default Risk"):
    input_df = pd.DataFrame([{
        "credit_lines_outstanding": credit_lines,
        "loan_amt_outstanding": loan_amt,
        "total_debt_outstanding": total_debt,
        "income": income,
        "years_employed": years_employed,
        "fico_score": fico_score,
    }])

    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0, 1]

    st.markdown("---")
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Default Probability", value=f"{probability * 100:.1f}%")
    with col2:
        if probability > 0.5:
            st.markdown(
                '<span style="background-color:#ff4b4b;color:white;padding:6px 18px;'
                'border-radius:8px;font-weight:bold;font-size:1.2em;">HIGH RISK</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span style="background-color:#21c354;color:white;padding:6px 18px;'
                'border-radius:8px;font-weight:bold;font-size:1.2em;">LOW RISK</span>',
                unsafe_allow_html=True,
            )

    st.progress(min(probability, 1.0))
    st.caption(f"Active model: **{active_model}**")

