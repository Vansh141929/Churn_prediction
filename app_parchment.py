import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Parchment Churn App", layout="wide", initial_sidebar_state="collapsed")

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "models/churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
FEATURES = model.get_booster().feature_names

st.markdown("""
<style>
:root {
  /* Colors */
  --color-parchment: #fefffc;
  --color-paper: #ffffff;
  --color-linen: #f9faf7;
  --color-ink-black: #171717;
  --color-graphite: #2c2c2c;
  --color-charcoal: #444141;
  --color-ash: #646464;
  --color-fog: #b4b8b4;
  --color-mist: #dee2de;
  --color-twilight: #282834;
  --color-dusk: #1f1f29;
  --color-signal-blue: #41a1cf;
  --color-cerulean: #0081c0;

  /* Typography — Font Families */
  --font-ppmondwest: 'ppmondwest', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-af: 'af', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;

  /* Typography — Scale */
  --text-caption: 13px;
  --leading-caption: 1.3;
  --tracking-caption: -0.13px;
  --text-body-sm: 15px;
  --leading-body-sm: 1;
  --tracking-body-sm: -0.15px;
  --text-subheading: 18px;
  --leading-subheading: 1.3;
  --tracking-subheading: -0.18px;
  --text-heading-sm: 27px;
  --leading-heading-sm: 1.5;
  --tracking-heading-sm: -1.08px;
  --text-heading: 40px;
  --leading-heading: 1.1;
  --tracking-heading: -0.8px;
  --text-heading-lg: 48px;
  --leading-heading-lg: 1.1;
  --tracking-heading-lg: -0.96px;
  --text-display: 54px;
  --leading-display: 1.1;
  --tracking-display: -1.08px;

  /* Typography — Weights */
  --font-weight-regular: 400;
  --font-weight-medium: 500;
  --font-weight-semibold: 600;
  --font-weight-bold: 700;

  /* Spacing */
  --spacing-unit: 4px;
  --spacing-4: 4px;
  --spacing-8: 8px;
  --spacing-12: 12px;
  --spacing-16: 16px;
  --spacing-20: 20px;
  --spacing-24: 24px;
  --spacing-32: 32px;
  --spacing-40: 40px;
  --spacing-48: 48px;
  --spacing-64: 64px;
  --spacing-80: 80px;

  /* Border Radius */
  --radius-md: 4px;
  --radius-lg: 8px;
  --radius-xl: 12px;
  --radius-2xl: 16px;
  --radius-3xl: 24px;
  --radius-full: 50px;

  /* Named Radii */
  --radius-nav: 50px;
  --radius-cards: 16px;
  --radius-buttons: 8px;
  --radius-large-surfaces: 24px;

  /* Shadows */
  --shadow-sm: rgba(0, 0, 0, 0.15) 0px 2px 6px 0px;
  --shadow-subtle: rgba(0, 0, 0, 0.08) 0px 1px 1px 0px, rgba(0, 0, 0, 0.08) 0px 4px 5px 0px;
  --shadow-subtle-2: rgb(222, 226, 222) 0px 0px 0px 1px;
  --shadow-subtle-3: rgba(0, 0, 0, 0.06) 0px 2px 2px 0px, rgba(0, 0, 0, 0.04) 0px 0px 0px 5px;
  --shadow-sm-2: rgba(0, 0, 0, 0.05) 0px 1px 8px 0px;

  /* Surfaces */
  --surface-parchment-canvas: #fefffc;
  --surface-paper-card: #ffffff;
  --surface-linen-wash: #f9faf7;
  --surface-vivid-blue: #0081c0;
}

html, body, [class*="css"] {
    background-color: var(--surface-parchment-canvas) !important;
    font-family: var(--font-af) !important;
    color: var(--color-ink-black) !important;
}

#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Hero */
.hero-wordmark {
    font-family: var(--font-ppmondwest);
    font-size: var(--text-display);
    line-height: var(--leading-display);
    letter-spacing: var(--tracking-display);
    color: var(--color-ink-black);
    font-weight: var(--font-weight-bold);
    text-transform: capitalize;
    text-align: center;
    margin-top: var(--spacing-40);
    margin-bottom: var(--spacing-12);
}

.hero-sub {
    font-family: var(--font-af);
    font-size: var(--text-subheading);
    line-height: var(--leading-subheading);
    letter-spacing: var(--tracking-subheading);
    color: var(--color-charcoal);
    font-weight: var(--font-weight-medium);
    text-align: center;
    margin-bottom: var(--spacing-64);
}

.section-headline {
    font-family: var(--font-ppmondwest);
    font-size: var(--text-heading-sm);
    line-height: var(--leading-heading-sm);
    letter-spacing: var(--tracking-heading-sm);
    color: var(--color-ink-black);
    font-weight: var(--font-weight-semibold);
    margin-bottom: var(--spacing-20);
}

/* Cards */
.parchment-card {
    background-color: var(--surface-paper-card);
    border-radius: var(--radius-cards);
    padding: var(--spacing-24);
    box-shadow: var(--shadow-subtle), var(--shadow-subtle-2);
    height: 100%;
}

.tag-chip {
    display: inline-block;
    border-radius: var(--radius-full);
    padding: 6px 12px;
    font-family: var(--font-af);
    font-weight: var(--font-weight-semibold);
    font-size: var(--text-caption);
    background-color: var(--surface-linen-wash);
    color: var(--color-cerulean);
    border: 1px solid var(--color-mist);
    margin-bottom: var(--spacing-16);
}

/* Streamlit Inputs */
div[data-baseweb="select"] > div, .stSlider > div > div > div {
    background-color: var(--surface-parchment-canvas) !important;
    border: 1px solid var(--color-mist) !important;
    border-radius: var(--radius-md) !important;
    color: var(--color-ink-black) !important;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.02);
}

div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label {
    font-family: var(--font-af) !important;
    font-size: var(--text-body-sm) !important;
    line-height: var(--leading-body-sm) !important;
    color: var(--color-graphite) !important;
    font-weight: var(--font-weight-medium) !important;
    margin-bottom: var(--spacing-8) !important;
}

div[data-baseweb="popover"] {
    background-color: var(--surface-paper-card) !important;
    color: var(--color-ink-black) !important;
    border: 1px solid var(--color-mist) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm);
}

/* Primary CTA Button */
.stButton > button {
    background-color: var(--surface-vivid-blue) !important;
    color: var(--color-paper) !important;
    font-family: var(--font-af) !important;
    font-size: var(--text-subheading);
    font-weight: var(--font-weight-semibold);
    padding: var(--spacing-16) var(--spacing-32);
    border-radius: var(--radius-buttons) !important;
    border: none;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
    margin-top: var(--spacing-24);
}
.stButton > button:hover {
    background-color: var(--color-signal-blue) !important;
    box-shadow: var(--shadow-sm-2);
}

.result-val {
    font-family: var(--font-ppmondwest);
    font-size: var(--text-heading-lg);
    line-height: var(--leading-heading-lg);
    letter-spacing: var(--tracking-heading-lg);
    color: var(--color-cerulean);
    font-weight: var(--font-weight-bold);
    margin-top: var(--spacing-12);
}

hr {
    border-color: var(--color-mist);
    margin: var(--spacing-64) 0;
}
</style>
""", unsafe_allow_html=True)

# Main Hero
st.markdown("""
<div class='hero-wordmark'>Customer Intelligence</div>
<div class='hero-sub'>Predictive analytics on a pristine parchment canvas</div>
""", unsafe_allow_html=True)

st.markdown("<div class='section-headline'>Customer Profile</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='parchment-card'><span class='tag-chip'>Data</span>", unsafe_allow_html=True)
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='parchment-card'><span class='tag-chip'>Finance</span>", unsafe_allow_html=True)
    monthly_charges = st.slider("Monthly Charges (₹)", 10, 200, 70)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='parchment-card'><span class='tag-chip'>Billing</span>", unsafe_allow_html=True)
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    total_charges = tenure * monthly_charges
    st.markdown(f"<div style='margin-top: {st.session_state.get('margin', '20px')};'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family: var(--font-af); color: var(--color-ash); font-size: var(--text-body-sm); font-weight: var(--font-weight-medium);'>Lifetime Value</div><div style='font-family: var(--font-ppmondwest); font-size: var(--text-heading-sm); font-weight: var(--font-weight-bold); color: var(--color-ink-black); margin-top: var(--spacing-8);'>₹{total_charges:.2f}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='section-headline'>Service Signals</div>", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown("<div class='parchment-card'><span class='tag-chip'>Support</span>", unsafe_allow_html=True)
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)
with s2:
    st.markdown("<div class='parchment-card'><span class='tag-chip'>Add-ons</span>", unsafe_allow_html=True)
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)
with s3:
    st.markdown("<div class='parchment-card'><span class='tag-chip'>Ops</span>", unsafe_allow_html=True)
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
st.markdown("<div class='section-headline'>Generation Output</div>", unsafe_allow_html=True)

if st.button("Calculate Risk"):
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
        <div class='parchment-card'>
            <div style='font-family: var(--font-af); color: var(--color-ash); font-size: var(--text-body-sm); font-weight: var(--font-weight-medium);'>Calibrated Probability</div>
            <div class='result-val'>{model_prob:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class='parchment-card'>
            <div style='font-family: var(--font-af); color: var(--color-ash); font-size: var(--text-body-sm); font-weight: var(--font-weight-medium);'>Business Risk Score</div>
            <div class='result-val'>{risk_score:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if risk_score >= 0.65:
        st.error("HIGH RISK — Action required.")
    elif risk_score >= 0.40:
        st.warning("MODERATE RISK — Monitor actively.")
    else:
        st.success("BASELINE — Stable.")
