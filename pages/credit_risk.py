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
    Choose to either upload a CSV file for batch prediction or manually enter a single record for prediction.
""")

# Option for dataset upload or single record input (default is Batch CSV Upload)
upload_option = st.selectbox(
    "Choose how you'd like to input data:",
    ("Batch CSV Upload", "Single Record"),
    index=0  # Default to Batch CSV Upload
)

# Provide CSV template for download
def download_template():
    # Creating a blank template DataFrame with the required columns
    template_data = {
        'income': [None],
        'credit_score': [None],
        'debt_to_income_ratio': [None],
        'loan_amount_requested': [None],
        'age': [None],
        'num_of_credit_inquiries': [None],
        'loan_amount': [None],
        'loan_term': [None],
        'interest_rate': [None],
        'credit_history_length': [None],
        'collateral_value': [None],
        'loan_to_value_ratio': [None],
        'current_balance': [None],
        'employment_status': [None],  # Categorical
        'loan_type': [None],  # Categorical
        'education_level': [None],  # Categorical
        'marital_status': [None],  # Categorical
        'repayment_schedule': [None]  # Categorical
    }
    template_df = pd.DataFrame(template_data)

    # Export template to CSV format
    csv = template_df.to_csv(index=False).encode('utf-8')

    # Allow download of the CSV template
    st.download_button(
        label="Download CSV Template",
        data=csv,
        file_name='credit_risk_template.csv',
        mime='text/csv',
    )

# Function to process a single record
def process_single_record():
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
    current_balance = st.slider('Current Balance', min_value=50.0, value=13000.0)

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

    return input_data

# Function to add risk and advice based on the prediction
def add_risk_and_advice(data):
    risk = []
    advice = []
    factors = []

    for index, row in data.iterrows():
        risk_level = ""
        advice_text = ""
        risk_factors = []
        
        probability = row["Probability of Default"]
        
        if probability < 0.33:
            risk_level = "LOW RISK of default"
            advice_text = "The applicant has a low probability of default. Consider this application favorably."
        elif 0.33 <= probability < 0.66:
            risk_level = "MEDIUM RISK of default"
            advice_text = ("The applicant presents a medium risk. Review their financials further, "
                          "especially credit score, loan amount, and debt-to-income ratio.")
        else:
            risk_level = "HIGH RISK of default"
            advice_text = ("The applicant poses a high risk of default. It is recommended to either reject the application "
                           "or apply stricter terms such as requesting more collateral or higher interest rates.")

        # Analyzing risk factors
        if row['credit_score'] < 600:
            risk_factors.append("Low credit score")
        if row['debt_to_income_ratio'] > 0.4:
            risk_factors.append("High debt-to-income ratio")
        if row['loan_amount_requested'] > row['income'] * 0.5:
            risk_factors.append("High loan amount relative to income")
        if row['num_of_credit_inquiries'] > 5:
            risk_factors.append("Multiple credit inquiries")
        
        risk.append(risk_level)
        advice.append(advice_text)
        factors.append(", ".join(risk_factors) if risk_factors else "No significant risk factors")
    
    data['Risk Level'] = risk
    data['Advice'] = advice
    data['Risk Factors'] = factors

    return data

# If user selects "Batch CSV Upload"
if upload_option == "Batch CSV Upload":
    # Show CSV template download option
    st.write("### Download CSV Template")
    download_template()

    # Upload CSV for Batch Prediction
    st.write("### Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Perform encoding and add missing columns for prediction
        data_encoded = pd.get_dummies(data, columns=['employment_status', 'loan_type', 'education_level', 'marital_status', 'repayment_schedule'], drop_first=True)
        
        # Replace this list with the actual columns your model was trained on
        expected_columns = [
            'income', 'credit_score', 'debt_to_income_ratio', 'loan_amount_requested',
            'age', 'num_of_credit_inquiries', 'loan_term', 'interest_rate', 'credit_history_length',
            'collateral_value', 'loan_to_value_ratio', 'current_balance',
            'employment_status', 'loan_type', 'education_level', 'marital_status',
            'repayment_schedule', 'current_balance', 'loan_amount'
        ]
        
        # Add missing columns with zeros if they don't exist in the input data
        for col in expected_columns:
            if col not in data_encoded.columns:
                data_encoded[col] = 0

        # Ensure the columns are in the same order as the model expects
        data_encoded = data_encoded[expected_columns]

        # Perform predictions for the entire dataset
        try:
            probabilities = model.predict_proba(data_encoded)[:, 1]
            data['Probability of Default'] = probabilities
            data = add_risk_and_advice(data)

            # Display first 5 records
            st.write("### First five records with predictions:")
            st.write(data.head())

            # Option to download the results as CSV
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv", key='download-csv')

        except Exception as e:
            st.write(f"Error: {e}")

# If user selects "Single Record"
elif upload_option == "Single Record":
    st.write("### Enter a Single Record for Prediction")
    input_data = process_single_record()

    # Encode categorical variables, predict, and display the result
    if st.button('Predict'):
        input_data_encoded = pd.get_dummies(input_data, columns=['employment_status', 'loan_type', 'education_level', 'marital_status', 'repayment_schedule'], drop_first=True)
        
        # Replace this list with the actual columns your model was trained on
        expected_columns = [
            'income', 'credit_score', 'debt_to_income_ratio', 'loan_amount_requested',
            'age', 'num_of_credit_inquiries', 'loan_term', 'interest_rate', 'credit_history_length',
            'collateral_value', 'loan_to_value_ratio', 'current_balance',
            'employment_status', 'loan_type', 'education_level', 'marital_status',
            'repayment_schedule', 'current_balance', 'loan_amount'
        ]

        # Add missing columns with zeros if they don't exist in the input data
        for col in expected_columns:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0
        input_data_encoded = input_data_encoded[expected_columns]
        
        # Perform prediction for the single record
        try:
            probability = model.predict_proba(input_data_encoded)[:, 1]
            input_data['Probability of Default'] = probability
            input_data = add_risk_and_advice(input_data)

            st.write(input_data)

        except Exception as e:
            st.write(f"Error: {e}")
