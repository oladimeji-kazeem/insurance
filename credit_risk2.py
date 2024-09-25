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

# Define the input fields for the app (assuming these are part of the 34 features)
income = st.number_input('Income', min_value=1000, value=50000)
credit_score = st.slider('Credit Score', min_value=300, max_value=850, value=650)
employment_status = st.selectbox('Employment Status', ['Employed', 'Unemployed', 'Self-employed', 'Retired'])
loan_amount_requested = st.number_input('Loan Amount Requested', min_value=1000, value=10000)
debt_to_income_ratio = st.slider('Debt-to-Income Ratio', min_value=0.0, max_value=1.0, value=0.3)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
num_of_credit_inquiries = st.number_input('Number of Credit Inquiries', min_value=0, max_value=10, value=2)
loan_term = st.number_input('Loan Term (in months)', min_value=1, max_value=360, value=60)
interest_rate = st.slider('Interest Rate (%)', min_value=0.0, max_value=100.0, value=5.0)
credit_history_length = st.slider('Credit History Length', min_value=0.0, max_value=100.0, value=10.0)
loan_amount = st.number_input('Loan Amount', min_value=1000, value=10000)
collateral_value = st.number_input('Collateral Value', min_value=10.0, value=5000.0)
loan_to_value_ratio = st.slider('Loan To Value Ratio', min_value=0.0, max_value=1.0, value=0.1)
current_balance = st.number_input('Current Balance', min_value=50.0, value=13000.0)

# Add any other assumed categorical or numerical variables
loan_type = st.selectbox('Loan Type', ['Personal', 'Mortgage', 'Auto', 'Business'])
education_level = st.selectbox('Education Level', ['High School', "Bachelor's", "Master's", 'PhD'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Widowed'])
repayment_schedule = st.selectbox('Repayment Schedule', ['Monthly', 'Quarterly', 'Annually'])

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'income': [income],
    'credit_score': [credit_score],
    'debt_to_income_ratio': [debt_to_income_ratio],
    'loan_amount_requested': [loan_amount_requested],
    'age': [age],
    'num_of_credit_inquiries': [num_of_credit_inquiries],
    'loan_amount': [loan_amount],
    'loan_term': [loan_term],
    'interest_rate': [interest_rate],
    'credit_history_length': [credit_history_length],
    'collateral_value': [collateral_value],
    'loan_to_value_ratio': [loan_to_value_ratio],
    'current_balance': [current_balance],
    'employment_status': [employment_status],
    'loan_type': [loan_type],
    'education_level': [education_level],
    'marital_status': [marital_status],
    'repayment_schedule': [repayment_schedule]
})

# One-Hot Encode Categorical Variables
input_data_encoded = pd.get_dummies(input_data, columns=['employment_status', 'loan_type', 'education_level', 'marital_status', 'repayment_schedule'], drop_first=True)

# List of all the features the model was trained on (including the one-hot encoded columns)
expected_columns = [
    'income', 'credit_score', 'debt_to_income_ratio', 'loan_amount_requested',
    'age', 'num_of_credit_inquiries', 'loan_term', 'interest_rate', 'credit_history_length',
    'collateral_value', 'loan_to_value_ratio', 'current_balance',
    'employment_status', 'loan_type', 'education_level', 'marital_status',
    'repayment_schedule', 'current_balance', 'loan_amount'
]

# Add any missing columns from the current input, and set the value to 0 for those missing categories
for col in expected_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Ensure the order of columns is the same as expected by the model
input_data_encoded = input_data_encoded[expected_columns]

# Prediction logic
if st.button('Predict'):
    try:
        prediction = model.predict(input_data_encoded)
        probability = model.predict_proba(input_data_encoded)[:, 1]  # Probability of default

        # Classify based on the probability of default and style output accordingly
        if probability[0] < 0.33:
            risk = "LOW RISK of default"
            color = "green"
            advice = "The applicant has a low probability of default. Proceed with caution but consider this application favorably."
        elif 0.33 <= probability[0] < 0.66:
            risk = "MEDIUM RISK of default"
            color = "yellow"
            advice = ("The applicant presents a medium risk. Consider reviewing their financials further, particularly "
                      "their credit score, loan amount, and debt-to-income ratio. Further collateral may reduce risk.")
        else:
            risk = "HIGH RISK of default"
            color = "red"
            advice = ("The applicant poses a high risk of default. It is advised to either reject this application or "
                      "request additional collateral and apply higher interest rates to mitigate risk.")

        # Display the prediction results with colored output
        st.markdown(f"<h3 style='color:{color};'>The model predicts: {risk}</h3>", unsafe_allow_html=True)
        st.write(f"Probability of Default: {probability[0]:.2%}")
        
        # Risk factor insights
        st.write("### Risk Factors Influencing This Prediction:")
        if credit_score < 600:
            st.write("- **Low credit score**: The applicant's credit score is below the standard threshold, which increases their risk.")
        if debt_to_income_ratio > 0.4:
            st.write("- **High debt-to-income ratio**: A debt-to-income ratio higher than 40% can make it difficult for the applicant to manage repayments.")
        if loan_amount_requested > income * 0.5:
            st.write("- **High loan amount relative to income**: The requested loan amount is more than half of the applicant's annual income, which poses additional risk.")
        if num_of_credit_inquiries > 5:
            st.write("- **Multiple credit inquiries**: More than 5 recent credit inquiries could indicate financial instability.")
        
        # Provide advice to the credit manager
        st.write("### Advice for the Credit Manager:")
        st.write(advice)

    except Exception as e:
        st.write(f"Error during prediction: {e}")