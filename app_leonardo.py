import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Leonardo Churn Arena", layout="wide", initial_sidebar_state="collapsed")

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "models/churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
FEATURES = model.get_booster().feature_names

st.markdown("""
<style>
:root {
  /* Colors */
  --color-midnight-canvas: #000000;
  --color-obsidian-surface: #0a0a0a;
  --color-charcoal-card: #353535;
  --color-fog: #f5f5f5;
  --color-bone-white: #ffffff;
  --color-mist: #e5e5e5;
  --color-ash-text: #999999;
  --color-charcoal-mute: #666666;
  --color-ink-black: #030303;
  --color-leonardo-violet: #6e60ee;
  --color-toxic-lime: #03e65b;
  --color-voltage-yellow: #ffc533;
  --color-shock-pink: #ff3386;
  --color-ember-coral: #ff5d4b;
  --color-plasma-pink: #d25fff;
  --color-arc-blue: #33d0ff;

  /* Typography — Font Families */
  --font-leosans: 'leoSans', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-canvasans: 'canvaSans', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-ufficiodisplay: 'ufficioDisplay', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;

  /* Typography — Scale */
  --text-caption: 10px;
  --leading-caption: 1.2;
  --text-body: 14px;
  --leading-body: 1.33;
  --text-subheading: 19px;
  --leading-subheading: 1.15;
  --text-heading-sm: 34px;
  --leading-heading-sm: 0.92;
  --tracking-heading-sm: -0.68px;
  --text-heading: 48px;
  --leading-heading: 0.9;
  --tracking-heading: -0.96px;
  --text-heading-lg: 78px;
  --leading-heading-lg: 0.85;
  --tracking-heading-lg: -1.56px;
  --text-display: 98px;
  --leading-display: 0.8;
  --tracking-display: -1.96px;
  --text-display-xl: 165px;
  --leading-display-xl: 0.8;
  --tracking-display-xl: -3.3px;

  /* Typography — Weights */
  --font-weight-regular: 400;
  --font-weight-medium: 500;
  --font-weight-bold: 700;
  --font-weight-extrabold: 800;
  --font-weight-black: 900;

  /* Spacing */
  --spacing-5: 5px;
  --spacing-7: 7px;
  --spacing-10: 10px;
  --spacing-14: 14px;
  --spacing-17: 17px;
  --spacing-20: 20px;
  --spacing-27: 27px;
  --spacing-34: 34px;
  --spacing-41: 41px;
  --spacing-44: 44px;
  --spacing-47: 47px;
  --spacing-54: 54px;
  --spacing-61: 61px;
  --spacing-68: 68px;
  --spacing-135: 135px;
  --spacing-203: 203px;

  /* Layout */
  --page-max-width: 1440px;
  --section-gap: 80px;
  --card-padding: 20px;
  --element-gap: 10px;

  /* Border Radius */
  --radius-md: 6px;
  --radius-lg: 8.4375px;
  --radius-2xl: 16.4375px;
  --radius-2xl-2: 20.4375px;
  --radius-full: 60px;
  --radius-full-2: 270px;

  /* Named Radii */
  --radius-tags: 270px;
  --radius-cards: 8px;
  --radius-inputs: 6px;
  --radius-buttons: 60px;
  --radius-cardslarge: 20px;
  --radius-imagecards: 8px;

  /* Surfaces */
  --surface-midnight-canvas: #000000;
  --surface-obsidian-surface: #0a0a0a;
  --surface-charcoal-card: #353535;
}

html, body, [class*="css"] {
    background-color: var(--color-midnight-canvas) !important;
    font-family: var(--font-canvasans) !important;
    color: var(--color-bone-white) !important;
}

#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Hero Signature */
.hero-wordmark {
    font-family: var(--font-leosans);
    font-size: clamp(60px, 10vw, var(--text-display-xl));
    line-height: var(--leading-display-xl);
    letter-spacing: var(--tracking-display-xl);
    color: var(--color-leonardo-violet);
    font-weight: 900;
    text-transform: uppercase;
    text-align: center;
    margin-top: 40px;
    margin-bottom: 20px;
}

.hero-sub {
    font-family: var(--font-canvasans);
    font-size: var(--text-heading-sm);
    line-height: var(--leading-heading-sm);
    letter-spacing: var(--tracking-heading-sm);
    color: var(--color-bone-white);
    font-weight: 800;
    text-align: center;
    text-transform: uppercase;
    margin-bottom: 80px;
}

.section-headline {
    font-family: var(--font-leosans);
    font-size: var(--text-heading);
    line-height: var(--leading-heading);
    letter-spacing: var(--tracking-heading);
    color: var(--color-bone-white);
    font-weight: 800;
    margin-bottom: 10px;
}

/* Cards (Editorial Grid) */
.leo-card {
    background-color: var(--color-charcoal-card);
    border-radius: var(--radius-cards);
    padding: 20px;
    box-shadow: 0 0 0 1px rgba(229,229,229,0.06), 0 20px 40px rgba(0,0,0,0.5);
    height: 100%;
}

.tag-chip {
    display: inline-block;
    border-radius: var(--radius-tags);
    padding: 7px 14px;
    font-family: var(--font-canvasans);
    font-weight: 700;
    font-size: 12px;
    color: var(--color-bone-white);
    margin-bottom: 10px;
}

/* Streamlit Inputs */
div[data-baseweb="select"] > div, .stSlider > div > div > div {
    background-color: transparent !important;
    border: 1px solid var(--color-mist) !important;
    border-radius: var(--radius-inputs) !important;
    color: var(--color-bone-white) !important;
}

div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label {
    font-family: var(--font-canvasans) !important;
    font-size: var(--text-body) !important;
    line-height: var(--leading-body) !important;
    color: var(--color-ash-text) !important;
    font-weight: 500 !important;
    margin-bottom: 10px !important;
}

div[data-baseweb="popover"] {
    background-color: var(--color-charcoal-card) !important;
    color: var(--color-bone-white) !important;
    border: 1px solid var(--color-mist) !important;
    border-radius: var(--radius-inputs) !important;
}

/* Primary CTA Button */
.stButton > button {
    background-color: var(--color-bone-white) !important;
    color: var(--color-obsidian-surface) !important;
    font-family: var(--font-canvasans) !important;
    font-size: 16px;
    font-weight: 500;
    padding: 10px 27px;
    border-radius: var(--radius-buttons) !important;
    border: none;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.08);
    transition: all 0.2s ease;
    margin-top: 20px;
}
.stButton > button:hover {
    background-color: var(--color-fog) !important;
}

.result-val {
    font-family: var(--font-leosans);
    font-size: var(--text-heading-lg);
    line-height: var(--leading-heading-lg);
    letter-spacing: var(--tracking-heading-lg);
    font-weight: 800;
    margin-top: 10px;
}

hr {
    border-color: rgba(229, 229, 229, 0.1);
    margin: 80px 0;
}
</style>
""", unsafe_allow_html=True)

# Main Hero
st.markdown("""
<div class='hero-wordmark'>LEONARDO</div>
<div class='hero-sub'>The Creator-First Generative AI Platform</div>
""", unsafe_allow_html=True)


st.markdown("<div class='section-headline'>Customer Profile</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='leo-card'><span class='tag-chip' style='background-color: var(--color-arc-blue);'>Data</span>", unsafe_allow_html=True)
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='leo-card'><span class='tag-chip' style='background-color: var(--color-shock-pink);'>Finance</span>", unsafe_allow_html=True)
    monthly_charges = st.slider("Monthly Charges (₹)", 10, 200, 70)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='leo-card'><span class='tag-chip' style='background-color: var(--color-ember-coral);'>Billing</span>", unsafe_allow_html=True)
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    total_charges = tenure * monthly_charges
    st.markdown(f"<div style='margin-top: {st.session_state.get('margin', '20px')};'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family: var(--font-canvasans); color: var(--color-ash-text); font-size: 14px; font-weight: 500;'>Lifetime Value</div><div style='font-family: var(--font-leosans); font-size: 34px; font-weight: 800; color: var(--color-bone-white);'>₹{total_charges:.2f}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='section-headline'>Service Signals</div>", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown("<div class='leo-card'><span class='tag-chip' style='background-color: var(--color-toxic-lime);'>Support</span>", unsafe_allow_html=True)
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)
with s2:
    st.markdown("<div class='leo-card'><span class='tag-chip' style='background-color: var(--color-voltage-yellow);'>Add-ons</span>", unsafe_allow_html=True)
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)
with s3:
    st.markdown("<div class='leo-card'><span class='tag-chip' style='background-color: var(--color-plasma-pink);'>Ops</span>", unsafe_allow_html=True)
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

if st.button("Start now"):
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
        <div class='leo-card'>
            <div style='font-family: var(--font-canvasans); color: var(--color-ash-text); font-size: 14px; font-weight: 500;'>Calibrated Probability</div>
            <div class='result-val' style='color: var(--color-bone-white);'>{model_prob:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class='leo-card'>
            <div style='font-family: var(--font-canvasans); color: var(--color-ash-text); font-size: 14px; font-weight: 500;'>Business Risk Score</div>
            <div class='result-val' style='color: var(--color-leonardo-violet);'>{risk_score:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if risk_score >= 0.65:
        st.error("HIGH RISK — Action required.")
    elif risk_score >= 0.40:
        st.warning("MODERATE RISK — Monitor actively.")
    else:
        st.success("BASELINE — Stable.")
