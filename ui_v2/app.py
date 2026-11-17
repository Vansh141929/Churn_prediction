import streamlit as st
import pandas as pd
import joblib
import os

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================
# LOAD MODEL & SCALER
# ======================================================
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "models/churn_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models/scaler.pkl")

# Fallback if running from subfolder
if not os.path.exists(model_path):
    BASE_DIR = os.path.dirname(BASE_DIR)
    model_path = os.path.join(BASE_DIR, "models/churn_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "models/scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
FEATURES = model.get_booster().feature_names

# ======================================================
# SLEEK MINIMALIST STYLING (MATCHING DESIGN)
# ======================================================
st.markdown("""
<style>
/* Base Dark Background */
html, body, [data-testid="stAppViewContainer"], .main {
    background-color: #0b0c10 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    color: #ffffff !important;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

#MainMenu, header, footer {
    visibility: hidden;
}

.block-container {
    padding-top: 3rem !important;
    padding-bottom: 5rem !important;
    max-width: 1200px !important;
}

/* Constellation top navigation glyphs */
.top-glyphs {
    font-size: 13px;
    color: #9333ea;
    letter-spacing: 16px;
    margin-bottom: 24px;
    font-weight: 700;
}

/* Monolithic Typography */
.title-giant {
    font-size: 82px;
    font-weight: 500;
    letter-spacing: -0.04em;
    line-height: 0.95;
    color: #ffffff;
    margin-bottom: 22px;
}

.title-large {
    font-size: 58px;
    font-weight: 500;
    letter-spacing: -0.04em;
    line-height: 1.0;
    color: #ffffff;
    margin-bottom: 16px;
}

.body-desc {
    font-size: 16px;
    color: #8b929d;
    line-height: 1.6;
    max-width: 580px;
}

.sub-desc {
    font-size: 15px;
    color: #717782;
    line-height: 1.5;
    max-width: 280px;
}

.accent-purple {
    color: #a855f7;
}

.accent-amber {
    color: #f59e0b;
}

/* Subtle Divider */
.custom-divider {
    border-top: 1px solid #1a1d24;
    margin-top: 50px;
    margin-bottom: 50px;
}

/* Form Controls Styling */
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label {
    font-weight: 500 !important;
    color: #717782 !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}

/* Selectbox Input Box */
div[data-baseweb="select"] > div {
    background-color: #171922 !important;
    border: 1px solid #232734 !important;
    border-radius: 7px !important;
    color: #ffffff !important;
    min-height: 42px !important;
}

div[data-baseweb="select"] svg {
    fill: #8b929d !important;
}

div[data-baseweb="popover"] {
    background-color: #171922 !important;
    border: 1px solid #232734 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

div[data-baseweb="popover"] li {
    color: #ffffff !important;
}

/* Slider value color */
div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
    color: #f87171 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

div[data-testid="stSlider"] [role="slider"] {
    background-color: #f87171 !important;
}

/* Pill Button */
.stButton > button {
    background: #8b5cf6 !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 14px 44px !important;
    border-radius: 50px !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.25) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.stButton > button:hover {
    background: #7c3aed !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(139, 92, 246, 0.4) !important;
    color: #ffffff !important;
}

/* Metric / Output Display */
.metric-row {
    font-size: 26px;
    font-weight: 400;
    color: #ffffff;
    margin-bottom: 12px;
}

.risk-alert {
    font-size: 16px;
    margin-top: 18px;
    font-weight: 400;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HERO SECTION
# ======================================================
st.markdown("<div class='top-glyphs'>▲ ▲ ▲</div>", unsafe_allow_html=True)
st.markdown("<div class='title-giant'>churn<br>intelligence<span class='accent-purple'>.</span></div>", unsafe_allow_html=True)
st.markdown("<div class='body-desc'>Knowledge visualized as distributed intelligence. Rather than hierarchical data, we employ calibrated XGBoost models overlaid with actionable business risk scoring.</div>", unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# ======================================================
# CUSTOMER PROFILE SECTION
# ======================================================
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown("<div class='title-large'>customer<br>profile<span class='accent-amber'>.</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-desc'>Define the baseline metrics of the subscriber.</div>", unsafe_allow_html=True)

with col_right:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        tenure = st.slider("TENURE (MONTHS)", 0, 72, 12)
        contract = st.selectbox("CONTRACT TYPE", ["Month-to-month", "One year", "Two year"])
    with c2:
        monthly_charges = st.slider("MONTHLY CHARGES (₹)", 10, 200, 70)
        internet_service = st.selectbox("INTERNET SERVICE", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox(
            "PAYMENT METHOD",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    total_charges = tenure * monthly_charges
    st.markdown(f"<div style='margin-top: 24px; font-size: 15px; color: #8b929d;'>Estimated Lifetime Value: <span class='accent-amber' style='font-weight: 600;'>₹{total_charges:,.2f}</span></div>", unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# ======================================================
# SERVICE SIGNALS SECTION
# ======================================================
col_s_left, col_s_right = st.columns([1, 1.4], gap="large")

with col_s_left:
    st.markdown("<div class='title-large'>service<br>signals<span class='accent-purple'>.</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-desc'>Identify active engagements and product footprint.</div>", unsafe_allow_html=True)

with col_s_right:
    s1, s2 = st.columns(2, gap="medium")
    with s1:
        tech_support = st.selectbox("TECH SUPPORT", ["No", "Yes"])
        online_security = st.selectbox("ONLINE SECURITY", ["No", "Yes"])
        paperless_billing = st.selectbox("PAPERLESS BILLING", ["Yes", "No"])
    with s2:
        device_protection = st.selectbox("DEVICE PROTECTION", ["No", "Yes"])
        streaming_tv = st.selectbox("STREAMING TV", ["Yes", "No"])

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# ======================================================
# RISK ASSESSMENT SECTION
# ======================================================
col_r_left, col_r_right = st.columns([1, 1.4], gap="large")

with col_r_left:
    st.markdown("<div class='title-large'>risk<br>assessment<span class='accent-amber'>.</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-desc'>Initialize predictive models and overlay business logic.</div>", unsafe_allow_html=True)

with col_r_right:
    if st.button("Initialize"):
        data = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "AvgMonthlyCharge": total_charges / (tenure + 1) if tenure > 0 else monthly_charges,
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

        X_scaled = scaler.transform(input_df)
        model_prob = model.predict_proba(X_scaled)[0][1]

        # Business Risk Overlay
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

        st.markdown(f"<div class='metric-row' style='margin-top: 20px;'>Base probability <span class='accent-amber'>— {model_prob:.2%}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-row'>Business score <span class='accent-purple'>— {risk_score:.2%}</span></div>", unsafe_allow_html=True)

        if risk_score >= 0.65:
            st.markdown("<div class='risk-alert' style='color: #ef4444;'>▲ <strong>Action required.</strong> High potential for attrition. Immediate retention incentive recommended.</div>", unsafe_allow_html=True)
        elif risk_score >= 0.40:
            st.markdown("<div class='risk-alert' style='color: #f59e0b;'>▲ <strong>Monitor actively.</strong> Moderate potential for attrition. Proactive engagement suggested.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='risk-alert' style='color: #10b981;'>▲ <strong>Baseline stability.</strong> Customer health is optimal.</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
