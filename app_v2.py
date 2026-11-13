import streamlit as st
import pandas as pd
import joblib
import os

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================
# LOAD MODEL & SCALER
# ======================================================
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "models/churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
FEATURES = model.get_booster().feature_names

# ======================================================
# UI STYLING (VIBRANT & MODERN)
# ======================================================
st.markdown("""
<style>
/* Base Theme */
html, body, [class*="css"] {
    background-color: #faf9f6; /* Off-white background */
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: #111111;
}

/* Hide Streamlit default top header and footer */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Custom Top Navigation Bar */
.top-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 40px;
    background-color: transparent;
    margin-bottom: 40px;
}
.nav-logo {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.05em;
    color: #111;
}
.nav-links {
    display: flex;
    gap: 30px;
    align-items: center;
    font-size: 14px;
    font-weight: 500;
}
.nav-item {
    cursor: pointer;
}
.nav-item:after {
    content: " ▾";
    font-size: 10px;
}
.nav-item.no-arrow:after {
    content: "";
}
.nav-btn {
    background-color: #ff4522; /* Vibrant Orange-Red */
    color: white;
    padding: 10px 24px;
    border-radius: 30px;
    font-weight: 600;
    text-decoration: none;
    transition: background-color 0.2s;
}
.nav-btn:hover {
    background-color: #e63e1e;
}

/* Main Hero Title */
.hero-title {
    text-align: center;
    font-size: 72px;
    font-weight: 500;
    line-height: 1.1;
    letter-spacing: -0.04em;
    margin-top: 20px;
    margin-bottom: 60px;
    color: #111;
}

/* Metric Cards - Clean & Minimal */
.metric-card {
    background: #ffffff;
    border-radius: 24px;
    padding: 30px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    border: 1px solid #f0f0f0;
    height: 100%;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.06);
}
.metric-card h4 {
    color: #666;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 12px;
    letter-spacing: -0.01em;
}
.metric-card h2 {
    font-size: 32px;
    font-weight: 600;
    margin: 0 0 8px 0;
    letter-spacing: -0.02em;
    color: #111;
}
.metric-card p {
    color: #888;
    margin: 0;
    font-size: 15px;
}

/* Form Sections */
.section {
    background: #ffffff;
    border-radius: 32px;
    padding: 40px;
    margin-bottom: 40px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.04);
    border: 1px solid #f0f0f0;
}
.section h3 {
    font-size: 28px;
    font-weight: 500;
    letter-spacing: -0.02em;
    margin-bottom: 30px;
    color: #111;
}

/* Streamlit Input Overrides */
div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label {
    font-weight: 500 !important;
    color: #444 !important;
    font-size: 15px !important;
}

/* Action Button */
.stButton > button {
    width: 100%;
    background-color: #ff4522 !important;
    color: white !important;
    font-size: 20px;
    font-weight: 500;
    padding: 20px;
    border-radius: 40px; /* Pill shape */
    border: none;
    box-shadow: 0 10px 25px rgba(255, 69, 34, 0.25);
    transition: all 0.2s;
    letter-spacing: -0.01em;
}
.stButton > button:hover {
    background-color: #e63e1e !important;
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(255, 69, 34, 0.35);
}
.stButton > button:active {
    transform: translateY(0);
}

/* Custom info/alert boxes */
div[data-testid="stInfo"] { background-color: #f8f9fa; border-left-color: #ff4522; color: #111; border-radius: 12px; }
div[data-testid="stSuccess"] { background-color: #ecfdf5; border-left-color: #10b981; color: #065f46; border-radius: 12px; }
div[data-testid="stWarning"] { background-color: #fffbeb; border-left-color: #f59e0b; color: #92400e; border-radius: 12px; }
div[data-testid="stError"] { background-color: #fef2f2; border-left-color: #ef4444; color: #991b1b; border-radius: 12px; }

</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER & NAVIGATION
# ======================================================
st.markdown("""
<div class="top-nav">
    <div class="nav-logo">churn.ai</div>
    <div class="nav-links">
        <div class="nav-item">For analysts</div>
        <div class="nav-item">Who we serve</div>
        <div class="nav-item">Model library</div>
        <div class="nav-item no-arrow">Pricing</div>
        <div class="nav-item">Community</div>
        <a href="#" class="nav-btn">Try for free</a>
    </div>
</div>

<div class="hero-title">
    churn<br>intelligence
</div>
""", unsafe_allow_html=True)

# ======================================================
# KPI CARDS
# ======================================================
k1, k2, k3 = st.columns(3)

with k1:
    st.markdown("""
    <div class="metric-card">
        <h4>Objective</h4>
        <h2>Churn Reduction</h2>
        <p>Retention-driven decisions</p>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown("""
    <div class="metric-card">
        <h4>Model</h4>
        <h2>XGBoost</h2>
        <p>Calibrated probability</p>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown("""
    <div class="metric-card">
        <h4>Decision Layer</h4>
        <h2>Business Rules</h2>
        <p>Actionable risk scoring</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ======================================================
# CUSTOMER PROFILE
# ======================================================
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<h3>Customer Subscription Profile</h3>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with c2:
    monthly_charges = st.slider("Monthly Charges (₹)", 10, 200, 70)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with c3:
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

total_charges = tenure * monthly_charges
st.info(f"💰 **Lifetime Value (Auto-Computed): ₹{total_charges:.2f}**")
st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# SERVICE SIGNALS (DATASET-ALIGNED)
# ======================================================
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<h3>Service & Support Signals</h3>", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)

with s1:
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])

with s2:
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])

with s3:
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# FEATURE ENGINEERING (MODEL INPUT)
# ======================================================
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

# ======================================================
# PREDICTION + BUSINESS RISK LAYER
# ======================================================
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<h3>Churn Risk Assessment</h3>", unsafe_allow_html=True)

if st.button("Run AI Risk Analysis"):
    X_scaled = scaler.transform(input_df)
    model_prob = model.predict_proba(X_scaled)[0][1]

    # ---------- BUSINESS RISK LAYER ----------
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

    # ---------- DISPLAY ----------
    st.markdown(f"**Model Probability:** {model_prob:.2%}")
    st.markdown(f"**Business Risk Score:** {risk_score:.2%}")

    if risk_score >= 0.65:
        st.error("🚨 **HIGH CHURN RISK — Immediate retention action required**")
    elif risk_score >= 0.40:
        st.warning("⚠️ **MODERATE CHURN RISK — Monitor & engage**")
    else:
        st.success("✅ **BASELINE CHURN RISK — Customer relatively stable**")

st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("<div style='text-align: center; color: #999; font-size: 13px; margin: 40px 0;'>Model probability + business logic = production-grade decision system</div>", unsafe_allow_html=True)
