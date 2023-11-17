import streamlit as st
from keras.models import load_model
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = load_model('best_model.h5')
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to make predictions
def make_prediction(input_data):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    return "Churn" if prediction[0, 0] > 0.5 else "No Churn"

# Streamlit UI Components
st.title("Customer Churn Prediction Web App")

# Sidebar for user input
with st.sidebar:
    st.header("Input User Data")
    monthly_charges = st.slider("Monthly Charges", min_value=0.0, max_value=1000.0, step=1.0, value=0.0)
    
    paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"])
    senior_citizen = st.selectbox("Senior Citizen", options=["Yes", "No"])
    payment_method = st.selectbox("Payment Method", options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    multiple_lines = st.selectbox("Multiple Lines", options=["Yes", "No"])

    submit_button = st.button("Submit")

# Encoding
def encode_data(monthly_charges,paperless_billing, senior_citizen, payment_method, multiple_lines):
    return {
        'MonthlyCharges': 1 if monthly_charges == "Yes" else 0,
        'PaperlessBilling_encoded': 1 if paperless_billing == "Yes" else 0,
        'SeniorCitizen_encoded': 1 if senior_citizen == "Yes" else 0,
        'PaymentMethod_encoded': {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}[payment_method],
        'MultipleLines_encoded': 1 if multiple_lines == "Yes" else 0,
    }

# Displaying the prediction
if submit_button:
    user_data = pd.DataFrame([encode_data(monthly_charges,paperless_billing, senior_citizen, payment_method, multiple_lines)], 
                             columns=['MonthlyCharges','PaperlessBilling_encoded','SeniorCitizen','PaymentMethod_encoded', 'MultipleLines_encoded'])
    user_data['MonthlyCharges'] = monthly_charges
    prediction = make_prediction(user_data)
    st.subheader("Prediction")
    st.write(prediction)
