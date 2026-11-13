import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Dala Churn Intelligence", layout="wide", initial_sidebar_state="collapsed")

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "models/churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
FEATURES = model.get_booster().feature_names

st.markdown("""
<style>
/* Dala Dark Stage */
html, body, [class*="css"] {
    background-color: #050505 !important;
    font-family: 'Helvetica Neue', 'Arial', sans-serif !important;
    color: #ffffff !important;
}

/* Hide Streamlit default top header and footer */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Monolithic Typography */
h1, h2, h3, h4, h5, h6, .monolithic {
    font-weight: 400 !important;
    letter-spacing: -0.04em !important;
    color: #ffffff !important;
    margin: 0;
    line-height: 1.1;
}

.title-giant {
    font-size: 113px;
    margin-bottom: 20px;
    font-weight: 400;
    letter-spacing: -0.05em;
    line-height: 0.9;
}

.title-large {
    font-size: 78px;
    margin-bottom: 40px;
    font-weight: 400;
    letter-spacing: -0.05em;
    line-height: 1.0;
}

.body-text {
    font-size: 20px;
    color: #888888;
    line-height: 1.5;
    font-weight: 400;
}

.accent-violet {
    color: #8A2BE2; /* Violet accent */
}

.accent-amber {
    color: #FFB000; /* Amber sparks */
}

/* Layout - No panels or borders */
.stSlider > div > div > div, .stSelectbox > div > div > div {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid #333 !important;
    color: #fff !important;
    border-radius: 0 !important;
}

/* Streamlit widgets overrides */
div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label {
    font-weight: 400 !important;
    color: #666 !important;
    font-size: 12px !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
div[data-baseweb="select"] > div {
    background-color: transparent;
    border-top: none;
    border-left: none;
    border-right: none;
    border-bottom: 1px solid #333;
    border-radius: 0;
    color: #fff;
}
div[data-baseweb="popover"] {
    background-color: #111 !important;
    color: #fff !important;
    border: 1px solid #333;
}

/* Violet Pill Button */
.stButton > button {
    width: auto;
    min-width: 200px;
    background-color: #8A2BE2 !important;
    color: white !important;
    font-size: 16px;
    font-weight: 400;
    padding: 16px 40px;
    border-radius: 50px; /* Pill shape */
    border: none;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    letter-spacing: -0.01em;
}
.stButton > button:hover {
    background-color: #9d4edd !important;
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(138, 43, 226, 0.3);
}

hr {
    border-color: #1a1a1a;
    margin: 100px 0;
}

/* Constellation signature gesture placeholder */
.brain-constellation {
    font-size: 18px;
    color: #8A2BE2;
    letter-spacing: 5px;
    line-height: 1.5;
    margin-bottom: 40px;
    margin-top: 40px;
    white-space: pre;
    font-family: monospace;
}
.brain-constellation span {
    color: #FFB000;
}

/* Custom info/alert boxes without borders/panels */
div[data-testid="stInfo"], div[data-testid="stSuccess"], div[data-testid="stWarning"], div[data-testid="stError"] { 
    background-color: transparent; 
    border: none; 
    padding: 0;
}
</style>
""", unsafe_allow_html=True)

# Main Hero
st.markdown("""
<div class='brain-constellation'>
  ▲  <span style='color:#FFB000'>▲</span>  ▲
▲ <span style='color:#FFB000'>▲</span>  ▲  <span style='color:#FFB000'>▲</span> ▲
  ▲  ▲  ▲
</div>
""", unsafe_allow_html=True)
st.markdown("<div class='title-giant'>churn<br>intelligence<span class='accent-violet'>.</span></div>", unsafe_allow_html=True)
st.markdown("<div class='body-text' style='margin-bottom: 40px; max-width: 600px;'>Knowledge visualized as distributed intelligence. Rather than hierarchical data, we employ calibrated XGBoost models overlaid with actionable business risk scoring.</div>", unsafe_allow_html=True)

# Two-column rhythm for Customer Profile
st.markdown("---")
col_left, col_right = st.columns([1, 1.2])
with col_left:
    st.markdown("<div class='title-large'>customer<br>profile<span class='accent-amber'>.</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='body-text' style='max-width: 300px;'>Define the baseline metrics of the subscriber.</div>", unsafe_allow_html=True)
with col_right:
    c1, c2 = st.columns(2)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with c2:
        monthly_charges = st.slider("Monthly Charges (₹)", 10, 200, 70)
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
    total_charges = tenure * monthly_charges
    st.markdown(f"<div class='body-text' style='margin-top: 40px;'>Lifetime Value: <span class='accent-amber'>₹{total_charges:.2f}</span></div>", unsafe_allow_html=True)

# Two-column rhythm for Service Signals
st.markdown("---")
col_left2, col_right2 = st.columns([1, 1.2])
with col_left2:
    st.markdown("<div class='title-large'>service<br>signals<span class='accent-violet'>.</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='body-text' style='max-width: 300px;'>Identify active engagements and product footprint.</div>", unsafe_allow_html=True)
with col_right2:
    s1, s2 = st.columns(2)
    with s1:
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    with s2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])

# Logic
data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "AvgMonthlyCharge": total_charges / (tenure + 1),
    "HighTenure": 1 if tenure > 24 else 0,

    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,

    "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
    "InternetService_No": 1 if internet_service == "No" else 0,

    "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
    "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,

    "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
    "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,
    "DeviceProtection_Yes": 1 if device_protection == "Yes" else 0,
    "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,
    "PaperlessBilling_Yes": 1 if paperless_billing == "Yes" else 0,
}

input_df = pd.DataFrame([data])
for col in FEATURES:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[FEATURES]

st.markdown("---")
col_left3, col_right3 = st.columns([1, 1.2])
with col_left3:
    st.markdown("<div class='title-large'>risk<br>assessment<span class='accent-amber'>.</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='body-text' style='max-width: 300px;'>Initialize predictive models and overlay business logic.</div>", unsafe_allow_html=True)
with col_right3:
    if st.button("Initialize"):
        X_scaled = scaler.transform(input_df)
        model_prob = model.predict_proba(X_scaled)[0][1]

        risk_score = model_prob

        if tenure < 6:
            risk_score += 0.15
        if contract == "Month-to-month":
            risk_score += 0.15
        if tech_support == "No":
            risk_score += 0.20
        if online_security == "No":
            risk_score += 0.20
        if paperless_billing == "Yes":
            risk_score += 0.05

        risk_score = min(risk_score, 1.0)

        st.markdown(f"<div style='font-size: 32px; font-weight: 400; margin-top: 40px; margin-bottom: 10px; color: #fff;'>Base probability <span class='accent-amber'>— {model_prob:.2%}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 32px; font-weight: 400; margin-bottom: 30px; color: #fff;'>Business score <span class='accent-violet'>— {risk_score:.2%}</span></div>", unsafe_allow_html=True)

        if risk_score >= 0.65:
            st.markdown("<div class='body-text' style='color: #ff4d4d;'>Action required. High potential for attrition.</div>", unsafe_allow_html=True)
        elif risk_score >= 0.40:
            st.markdown("<div class='body-text' style='color: #FFB000;'>Monitor actively. Moderate potential for attrition.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='body-text' style='color: #a3a3a3;'>Baseline stability. Customer engaged.</div>", unsafe_allow_html=True)

st.markdown("<br><br><br><br>", unsafe_allow_html=True)
