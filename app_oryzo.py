import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="ORYZO Churn Artifact", layout="wide", initial_sidebar_state="collapsed")

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "models/churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
FEATURES = model.get_booster().feature_names

st.markdown("""
<style>
/* Warm Dark Canvas */
html, body, [class*="css"] {
    background-color: #161513 !important;
    font-family: 'Helvetica Neue', Arial, sans-serif !important;
    color: #e8e3d3 !important;
}

/* Hide Streamlit default UI chrome */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Uppercase Weight 500 Rule */
h1, h2, h3, h4, h5, h6, label, .uppercase-text {
    text-transform: uppercase !important;
    font-weight: 500 !important;
    color: #e8e3d3 !important;
    letter-spacing: 0.05em;
    margin: 0;
}

.title-giant {
    font-size: 72px;
    margin-bottom: 30px;
    margin-top: 20px;
    line-height: 1.1;
    text-align: center;
}

/* Body Copy Rule (System's only conversational voice: 29px/400) */
.body-conversational {
    text-transform: none !important;
    font-weight: 400 !important;
    font-size: 29px !important;
    color: #b5b0a1 !important;
    line-height: 1.4;
    text-align: center;
    max-width: 800px;
    margin: 0 auto 80px auto;
}

/* Pill-shaped controls */
div[data-baseweb="select"] > div, .stSlider > div > div > div {
    background-color: transparent !important;
    border: 1px solid #4a4740 !important;
    border-radius: 50px !important;
    color: #e8e3d3 !important;
    padding: 0 10px;
}
div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label {
    margin-bottom: 15px;
    font-size: 12px !important;
    letter-spacing: 0.1em;
}
div[data-baseweb="popover"] {
    background-color: #161513 !important;
    color: #e8e3d3 !important;
    border: 1px solid #4a4740 !important;
    border-radius: 20px !important;
}

/* Button - Pill-shaped, NO ORANGE */
.stButton {
    display: flex;
    justify-content: center;
}
.stButton > button {
    width: auto;
    min-width: 300px;
    background-color: transparent !important;
    color: #e8e3d3 !important;
    font-size: 14px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 24px 40px;
    border-radius: 50px !important;
    border: 1px solid #e8e3d3 !important;
    transition: all 0.3s ease;
    margin: 20px auto;
}
.stButton > button:hover {
    background-color: #e8e3d3 !important;
    color: #161513 !important;
    transform: translateY(-2px);
}

/* Hairline dashed dividers */
hr {
    border-color: transparent;
    border-top: 1px dashed #3a3730;
    margin: 80px 0;
}

.section-title {
    font-size: 14px;
    letter-spacing: 0.15em;
    margin-bottom: 60px;
    text-align: center;
    color: #8c887d !important;
}

/* Studio Link / Credits in Vivid Orange */
.studio-credit {
    color: #FF4500 !important;
    text-transform: uppercase;
    font-weight: 500;
    font-size: 12px;
    letter-spacing: 0.15em;
    text-align: center;
    margin-top: 120px;
    margin-bottom: 40px;
}

/* Custom info/alert boxes without borders */
div[data-testid="stInfo"], div[data-testid="stSuccess"], div[data-testid="stWarning"], div[data-testid="stError"] { 
    background-color: transparent; 
    border: none; 
    padding: 20px 0;
    text-align: center;
}
.result-val {
    font-size: 64px;
    font-weight: 500;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# Main Hero
st.markdown("<div class='title-giant uppercase-text'>CHURN ARTIFACT</div>", unsafe_allow_html=True)
st.markdown("<div class='body-conversational'>Isolated from the noise, the predictive models and rules evaluate the subscriber's baseline stability within the warm darkness.</div>", unsafe_allow_html=True)

# Layout Container
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='section-title uppercase-text'>1. CUSTOMER PROFILE</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
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
st.markdown(f"<div class='uppercase-text' style='text-align: center; margin-top: 70px; font-size: 14px; letter-spacing: 0.15em; color: #8c887d;'>LIFETIME VALUE <span style='color:#e8e3d3; margin-left: 10px;'>₹{total_charges:.2f}</span></div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='section-title uppercase-text'>2. SERVICE SIGNALS</div>", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)
with s1:
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
with s2:
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
with s3:
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

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

if st.button("EXECUTE ANALYSIS"):
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

    st.markdown("<br><br>", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"<div style='text-align: center;'><div class='uppercase-text' style='color: #8c887d; font-size: 14px; letter-spacing: 0.1em;'>CALIBRATED PROBABILITY</div><div class='result-val'>{model_prob:.2%}</div></div>", unsafe_allow_html=True)
    with r2:
        st.markdown(f"<div style='text-align: center;'><div class='uppercase-text' style='color: #8c887d; font-size: 14px; letter-spacing: 0.1em;'>BUSINESS RISK SCORE</div><div class='result-val'>{risk_score:.2%}</div></div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    if risk_score >= 0.65:
        st.markdown("<div class='uppercase-text' style='text-align: center; font-size: 18px; letter-spacing: 0.1em; color: #e8e3d3;'>STATUS: HIGH RISK <br> IMMEDIATE ACTION REQUIRED</div>", unsafe_allow_html=True)
    elif risk_score >= 0.40:
        st.markdown("<div class='uppercase-text' style='text-align: center; font-size: 18px; letter-spacing: 0.1em; color: #8c887d;'>STATUS: MODERATE RISK <br> MONITOR ACTIVELY</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='uppercase-text' style='text-align: center; font-size: 18px; letter-spacing: 0.1em; color: #8c887d;'>STATUS: BASELINE <br> RELATIVELY STABLE</div>", unsafe_allow_html=True)

st.markdown("<div class='studio-credit'>STUDIO ORYZO // CHURN PREDICTION SYSTEMS</div>", unsafe_allow_html=True)
