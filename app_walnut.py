import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Walnut Churn Predictor", layout="wide", initial_sidebar_state="collapsed")

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "models/churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
FEATURES = model.get_booster().feature_names

st.markdown("""
<style>
:root {
  /* Colors */
  --color-warm-cream: #ffedd7;
  --color-walnut-shadow: #100904;
  --color-bark-brown: #382416;
  --color-cork-border: #40372e;
  --color-driftwood: #6c5f51;
  --color-ember-accent: #dc5000;
  --color-pure-black: #000000;

  /* Typography — Font Families */
  --font-halyard-display-variable: 'halyard-display-variable', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-arial: 'Arial', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;

  /* Typography — Scale */
  --text-subheading: 18px;
  --leading-subheading: 1;
  --text-heading-sm: 24px;
  --leading-heading-sm: 1.09;
  --text-body: 29px;
  --leading-body: 1.26;
  --text-heading: 41px;
  --leading-heading: 0.9;
  --text-display: 51px;
  --leading-display: 0.9;

  /* Typography — Weights */
  --font-weight-regular: 400;
  --font-weight-medium: 500;

  /* Spacing */
  --spacing-6: 6px;
  --spacing-8: 8px;
  --spacing-9: 9px;
  --spacing-10: 10px;
  --spacing-12: 12px;
  --spacing-14: 14px;
  --spacing-18: 18px;
  --spacing-24: 24px;
  --spacing-31: 31px;
  --spacing-41: 41px;
  --spacing-45: 45px;
  --spacing-68: 68px;
  --spacing-204: 204px;

  /* Layout */
  --card-padding: 24px;
  --element-gap: 18px;

  /* Border Radius */
  --radius-xl: 12px;
  --radius-2xl: 22.5px;
  --radius-3xl: 36px;
  --radius-full: 9999px;

  /* Named Radii */
  --radius-cards: 12px;
  --radius-inputs: 0px;
  --radius-full-round: 9999px;
  --radius-buttons-pill: 36px;
  --radius-buttons-outlined: 22.5px;

  /* Surfaces */
  --surface-walnut-shadow: #100904;
  --surface-bark-brown: #382416;
  --surface-cork-border: #40372e;
  --surface-warm-cream: #ffedd7;
}

/* Global Styles */
html, body, [class*="css"] {
    background-color: var(--surface-walnut-shadow) !important;
    font-family: var(--font-halyard-display-variable) !important;
    color: var(--color-warm-cream) !important;
}

#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Typography Classes */
.display-text {
    font-size: var(--text-display);
    line-height: var(--leading-display);
    font-weight: var(--font-weight-medium);
    margin-bottom: var(--spacing-24);
}
.heading-text {
    font-size: var(--text-heading);
    line-height: var(--leading-heading);
    font-weight: var(--font-weight-medium);
    margin-bottom: var(--spacing-18);
}
.body-text {
    font-size: var(--text-body);
    line-height: var(--leading-body);
    font-weight: var(--font-weight-regular);
    margin-bottom: var(--spacing-41);
    color: var(--color-warm-cream);
}
.accent {
    color: var(--color-ember-accent);
}

/* Cards */
.walnut-card {
    background-color: var(--surface-bark-brown);
    border-radius: var(--radius-cards);
    padding: var(--card-padding);
    border: 1px solid var(--surface-cork-border);
    margin-bottom: var(--element-gap);
    height: 100%;
}

.walnut-card-title {
    font-size: var(--text-heading-sm);
    line-height: var(--leading-heading-sm);
    color: var(--color-warm-cream);
    margin-bottom: var(--spacing-10);
}

.walnut-card-val {
    font-size: var(--text-display);
    color: var(--color-ember-accent);
    font-weight: var(--font-weight-medium);
}

/* Inputs */
div[data-baseweb="select"] > div, .stSlider > div > div > div {
    background-color: var(--surface-walnut-shadow) !important;
    border: 1px solid var(--surface-cork-border) !important;
    border-radius: var(--radius-inputs) !important;
    color: var(--color-warm-cream) !important;
}

div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label {
    font-size: var(--text-subheading) !important;
    line-height: var(--leading-subheading) !important;
    color: var(--color-warm-cream) !important;
    font-weight: var(--font-weight-regular) !important;
    margin-bottom: var(--spacing-6) !important;
}

div[data-baseweb="popover"] {
    background-color: var(--surface-bark-brown) !important;
    color: var(--color-warm-cream) !important;
    border: 1px solid var(--surface-cork-border) !important;
    border-radius: var(--radius-inputs) !important;
}

/* Pill Buttons */
.stButton > button {
    width: 100%;
    background-color: var(--color-warm-cream) !important;
    color: var(--surface-walnut-shadow) !important;
    font-size: var(--text-heading-sm);
    font-weight: var(--font-weight-medium);
    padding: var(--spacing-18) var(--spacing-41);
    border-radius: var(--radius-buttons-pill) !important;
    border: none;
    transition: all 0.2s ease;
    margin-top: var(--spacing-24);
}
.stButton > button:hover {
    background-color: var(--color-ember-accent) !important;
    color: var(--color-warm-cream) !important;
}

hr {
    border-color: var(--surface-cork-border);
    margin: var(--spacing-68) 0;
}
</style>
""", unsafe_allow_html=True)

# Main Hero
st.markdown("<div class='display-text'>Customer Churn <span class='accent'>Prediction</span></div>", unsafe_allow_html=True)
st.markdown("<div class='body-text'>Analyze subscriber baseline stability using advanced XGBoost modeling overlaid with actionable business logic.</div>", unsafe_allow_html=True)

st.markdown("<div class='heading-text'>Customer Profile</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='walnut-card'>", unsafe_allow_html=True)
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='walnut-card'>", unsafe_allow_html=True)
    monthly_charges = st.slider("Monthly Charges (₹)", 10, 200, 70)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='walnut-card'>", unsafe_allow_html=True)
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    total_charges = tenure * monthly_charges
    st.markdown(f"<div style='margin-top: {st.session_state.get('margin', '24px')};'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='walnut-card-title'>Lifetime Value</div><div class='walnut-card-val'>₹{total_charges:.2f}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='heading-text'>Service Signals</div>", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown("<div class='walnut-card'>", unsafe_allow_html=True)
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)
with s2:
    st.markdown("<div class='walnut-card'>", unsafe_allow_html=True)
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)
with s3:
    st.markdown("<div class='walnut-card'>", unsafe_allow_html=True)
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)

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

st.markdown("<hr>", unsafe_allow_html=True)

if st.button("Calculate Risk Assessment"):
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

    st.markdown("<br>", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"""
        <div class='walnut-card'>
            <div class='walnut-card-title'>Calibrated Probability</div>
            <div class='walnut-card-val' style='color: var(--color-warm-cream);'>{model_prob:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class='walnut-card'>
            <div class='walnut-card-title'>Business Risk Score</div>
            <div class='walnut-card-val'>{risk_score:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if risk_score >= 0.65:
        st.error("HIGH RISK: Immediate retention action required.")
    elif risk_score >= 0.40:
        st.warning("MODERATE RISK: Monitor actively.")
    else:
        st.success("BASELINE: Customer is relatively stable.")
