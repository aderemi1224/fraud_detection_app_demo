import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('rf_fraud_model.pkl')

st.title("💳 Fraud Detection Demo")
st.write("Enter transaction details:")

# Input fields
customer_id = st.text_input("Customer ID")
amount_usd = st.number_input("Transaction Amount (USD)", min_value=0.0, value=100.0)
fee = st.number_input("Transaction Fee", min_value=0.0, value=1.0)
device_trust_score = st.slider("Device Trust Score", 0.0, 1.0, 0.5)
risk_score_internal = st.slider("Internal Risk Score", 0.0, 1.0, 0.5)
account_age_days = st.number_input("Account Age (days)", min_value=0, value=365)
cust_avg_velocity_1h = st.number_input("Customer Avg Transactions Last 1h", min_value=0, value=0)
cust_avg_velocity_24h = st.number_input("Customer Avg Transactions Last 24h", min_value=0, value=0)
cust_avg_amount = st.number_input("Customer Avg Transaction Amount", min_value=0.0, value=100.0)
new_device = st.selectbox("New Device?", [0, 1])
corridor_risk = st.slider("Corridor Risk", 0.0, 1.0, 0.5)
amount_to_ratio_fee = st.number_input("Amount to Fee Ratio", min_value=0.0, value=100.0)
dest_currency = st.text_input("Destination Currency", value="USD")
channel = st.text_input("Transaction Channel", value="online")
ip_country = st.text_input("IP Country", value="US")
day = st.number_input("Day of Week (0=Monday, 6=Sunday)", min_value=0, max_value=6, value=0)

if st.button("Predict Fraud Score"):
    # Prepare input for model
    input_df = pd.DataFrame([{
        'amount_usd': amount_usd,
        'fee': fee,
        'device_trust_score': device_trust_score,
        'risk_score_internal': risk_score_internal,
        'account_age_days': account_age_days,
        'cust_avg_velocity_1h': cust_avg_velocity_1h,
        'cust_avg_velocity_24h': cust_avg_velocity_24h,
        'cust_avg_amount': cust_avg_amount,
        'new_device': new_device,
        'corridor_risk': corridor_risk,
        'amount_to_ratio_fee': amount_to_ratio_fee,
        'dest_currency': dest_currency,
        'channel': channel,
        'ip_country': ip_country,
        'day': day
    }])
    
    # Predict fraud probability
    score = model.predict_proba(input_df)[:,1][0]
    st.write(f"Fraud Score: **{score:.2f}** (0 = low risk, 1 = high risk)")
