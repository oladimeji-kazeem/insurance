import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load('model/default_probability_model.pkl')

# Title of the app
st.title("Credit Risk Prediction App")

# Instructions
st.write("""
    Enter the required information to predict the probability of default.
""")

# Define the input fields for the app
# Make sure the input fields match the features your model expects
income = st.number_input('Income', min_value=0, value=50000)
credit_score = st.slider('Credit Score', min_value=300, max_value=850, value=650)
employment_status = st.selectbox('Employment Status', ['Employed', 'Unemployed', 'Self-employed', 'Retired'])
loan_amount = st.number_input('Loan Amount Requested', min_value=0, value=10000)
debt_to_income_ratio = st.slider('Debt-to-Income Ratio', min_value=0.0, max_value=1.0, value=0.3)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
num_of_credit_inquiries = st.number_input('Number of Credit Inquiries', min_value=0, max_value=10, value=2)
loan_term = st.number_input('Loan Term (in months)', min_value=1, max_value=360, value=60)
interest_rate = st.slider('Interest Rate (%)', min_value=0.0, max_value=100.0, value=5.0)

# Prepare the input data for the model
# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'income': [income],
    'credit_score': [credit_score],
    'debt_to_income_ratio': [debt_to_income_ratio],
    'loan_amount': [loan_amount],
    'age': [age],
    'num_of_credit_inquiries': [num_of_credit_inquiries],
    'loan_term': [loan_term],
    'interest_rate': [interest_rate],
    'employment_status': [employment_status]
})

# One-Hot Encode Categorical Variable: Employment Status
# The model expects the same encoding as was done during training (create dummy variables)
input_data_encoded = pd.get_dummies(input_data, columns=['employment_status'], drop_first=True)

# List of all the features the model was trained on (including the one-hot encoded columns)
expected_columns = [
    'income', 'credit_score', 'debt_to_income_ratio', 'loan_amount',
    'age', 'num_of_credit_inquiries', 'loan_term', 'interest_rate',
    'employment_status_Employed', 'employment_status_Self-employed', 'employment_status_Retired'
]

# Add any missing columns from the current input, and set the value to 0 for those missing categories
for col in expected_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Ensure the order of columns is the same as expected by the model
input_data_encoded = input_data_encoded[expected_columns]

# Prediction logic
if st.button('Predict'):
    prediction = model.predict(input_data_encoded)
    probability = model.predict_proba(input_data_encoded)[:, 1]

    # Display the prediction results
    if prediction[0] == 1:
        st.write(f"### The model predicts: **HIGH RISK of default**")
        st.write(f"Probability of Default: {probability[0]:.2%}")
    else:
        st.write(f"### The model predicts: **LOW RISK of default**")
        st.write(f"Probability of Default: {probability[0]:.2%}")
