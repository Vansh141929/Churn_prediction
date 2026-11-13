import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Control Churn Platform", layout="wide", initial_sidebar_state="collapsed")

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "models/churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
FEATURES = model.get_booster().feature_names

st.markdown("""
<style>
:root {
  /* Colors */
  --color-whiteout: #ffffff;
  --color-haze: #f5f5f5;
  --color-ink: #1b1b1b;
  --color-black-void: #000000;
  --color-twilight-blue: #426188;
  --color-signal-blue: #2b7fff;

  /* Typography — Font Families */
  --font-control: 'Control', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-control-compressed: 'Control Compressed', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-control-cursive: 'Control Cursive', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-control-tnt: 'Control TNT', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;

  /* Typography — Scale */
  --text-caption: 13px;
  --leading-caption: 1.5;
  --text-body: 16px;
  --leading-body: 1.5;
  --text-subheading: 20px;
  --leading-subheading: 1.4;
  --text-heading: 32px;
  --leading-heading: 1.1;
  --text-heading-lg: 56px;
  --leading-heading-lg: 1;
  --text-display: 259px;
  --leading-display: 0.85;

  /* Typography — Weights */
  --font-weight-regular: 400;
  --font-weight-medium: 500;
  --font-weight-black: 900;

  /* Spacing */
  --spacing-unit: 4px;
  --spacing-4: 4px;
  --spacing-8: 8px;
  --spacing-12: 12px;
  --spacing-16: 16px;
  --spacing-20: 20px;
  --spacing-24: 24px;
  --spacing-32: 32px;
  --spacing-48: 48px;
  --spacing-52: 52px;
  --spacing-64: 64px;
  --spacing-80: 80px;
  --spacing-120: 120px;

  /* Layout */
  --page-max-width: 1150px;
  --section-gap: 48px;
  --card-padding: 20px;
  --element-gap: 8px;

  /* Border Radius */
  --radius-md: 4px;
  --radius-lg: 8px;
  --radius-lg-2: 11px;
  --radius-xl: 14px;

  /* Named Radii */
  --radius-cards: 12px;
  --radius-pills: 9999px;
  --radius-images: 11px;
  --radius-inputs: 4px;
  --radius-buttons: 8px;

  /* Surfaces */
  --surface-dark-canvas: #000000;
  --surface-haze-card: #f5f5f5;
  --surface-whiteout: #ffffff;
}

/* Global Layout */
html, body, [class*="css"] {
    background-color: var(--surface-dark-canvas) !important;
    font-family: var(--font-control) !important;
    color: var(--color-haze) !important;
}

#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Typography Classes */
.display-text {
    font-size: var(--text-display);
    line-height: var(--leading-display);
    font-weight: var(--font-weight-black);
    color: var(--color-ink);
    letter-spacing: -0.04em;
    position: absolute;
    top: -100px;
    left: 0;
    z-index: 0;
    white-space: nowrap;
    overflow: hidden;
}

.hero-wrapper {
    position: relative;
    padding-top: var(--spacing-80);
    padding-bottom: var(--spacing-48);
}

.heading-lg {
    font-family: var(--font-control-compressed);
    font-size: var(--text-heading-lg);
    line-height: var(--leading-heading-lg);
    font-weight: var(--font-weight-black);
    color: var(--color-whiteout);
    margin-bottom: var(--spacing-24);
    position: relative;
    z-index: 1;
    text-transform: uppercase;
}

.heading-text {
    font-size: var(--text-heading);
    line-height: var(--leading-heading);
    font-weight: var(--font-weight-black);
    color: var(--color-whiteout);
    margin-bottom: var(--spacing-24);
}

.body-text {
    font-size: var(--text-body);
    line-height: var(--leading-body);
    font-weight: var(--font-weight-regular);
    color: var(--color-haze);
    margin-bottom: var(--spacing-24);
    position: relative;
    z-index: 1;
}

/* Cards */
.control-card {
    background-color: var(--color-ink);
    border-radius: var(--radius-cards);
    padding: var(--card-padding);
    margin-bottom: var(--element-gap);
    border: 1px solid #2a2a2a;
    height: 100%;
}

/* Streamlit Inputs Overrides */
div[data-baseweb="select"] > div, .stSlider > div > div > div {
    background-color: var(--surface-dark-canvas) !important;
    border: 1px solid var(--color-twilight-blue) !important;
    border-radius: var(--radius-inputs) !important;
    color: var(--color-whiteout) !important;
}

div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label {
    font-family: var(--font-control-compressed) !important;
    font-size: var(--text-subheading) !important;
    line-height: var(--leading-subheading) !important;
    color: var(--color-haze) !important;
    font-weight: var(--font-weight-medium) !important;
    margin-bottom: var(--spacing-8) !important;
    text-transform: uppercase;
}

/* Base Web Dropdown */
div[data-baseweb="popover"] {
    background-color: var(--color-ink) !important;
    color: var(--color-whiteout) !important;
    border: 1px solid var(--color-twilight-blue) !important;
    border-radius: var(--radius-inputs) !important;
}

/* Action Button */
.stButton > button {
    width: 100%;
    background-color: var(--color-signal-blue) !important;
    color: var(--surface-whiteout) !important;
    font-family: var(--font-control-compressed) !important;
    font-size: var(--text-heading);
    font-weight: var(--font-weight-black);
    padding: var(--spacing-20) var(--spacing-48);
    border-radius: var(--radius-buttons) !important;
    border: none;
    transition: all 0.2s ease;
    text-transform: uppercase;
}
.stButton > button:hover {
    background-color: var(--color-twilight-blue) !important;
}

/* Custom layout spacing */
.section-spacer {
    margin-top: var(--section-gap);
    margin-bottom: var(--section-gap);
}

.val-display {
    font-size: var(--text-heading-lg);
    font-weight: var(--font-weight-black);
    color: var(--color-signal-blue);
    line-height: 1;
}

</style>
""", unsafe_allow_html=True)

# Main Hero
st.markdown("""
<div class='hero-wrapper'>
    <div class='display-text'>CHURN</div>
    <div class='heading-lg'>Predictive Architecture</div>
    <div class='body-text'>Operational risk scoring utilizing high-performance XGBoost inference.</div>
</div>
""", unsafe_allow_html=True)


st.markdown("<div class='heading-text'>CUSTOMER PROFILE</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
    monthly_charges = st.slider("Monthly Charges (₹)", 10, 200, 70)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    total_charges = tenure * monthly_charges
    st.markdown(f"<div style='margin-top: {st.session_state.get('margin', '24px')};'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family: var(--font-control-compressed); text-transform: uppercase; font-size: var(--text-subheading); color: var(--color-haze); font-weight: var(--font-weight-medium);'>Lifetime Value</div><div class='val-display'>₹{total_charges:.2f}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
st.markdown("<div class='heading-text'>SERVICE SIGNALS</div>", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)
with s2:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)
with s3:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
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

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

if st.button("RUN INFERENCE"):
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
        <div class='control-card'>
            <div style='font-family: var(--font-control-compressed); text-transform: uppercase; font-size: var(--text-subheading); font-weight: var(--font-weight-medium);'>Calibrated Probability</div>
            <div class='val-display' style='color: var(--color-whiteout);'>{model_prob:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class='control-card'>
            <div style='font-family: var(--font-control-compressed); text-transform: uppercase; font-size: var(--text-subheading); font-weight: var(--font-weight-medium);'>Business Risk Score</div>
            <div class='val-display' style='color: var(--color-signal-blue);'>{risk_score:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if risk_score >= 0.65:
        st.error("HIGH RISK — Immediate retention action required.")
    elif risk_score >= 0.40:
        st.warning("MODERATE RISK — Monitor actively.")
    else:
        st.success("BASELINE — Customer is relatively stable.")
