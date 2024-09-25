import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import io

# Function to generate CSV template
def generate_template():
    template_data = {
        'Customer_ID': [1001, 1002],
        'Age': [25, 40],
        'Gender': ['Male', 'Female'],
        'Income_Level': ['Medium', 'High'],
        'Occupation': ['Engineer', 'Teacher'],
        'Marital_Status': ['Single', 'Married'],
        'Education_Level': ['Bachelor', 'Master'],
        'Location': ['Urban', 'Suburban'],
        'Policy_Type': ['Life', 'Health'],
        'Coverage_Amount': [50000, 100000],
        'Premium_Amount': [1500, 2000],
        'Policy_Tenure_Years': [5, 7],
        'Claim_History': ['Yes', 'No'],
        'Payment_Preferences': ['Monthly', 'Annually'],
        'Customer_Support_Tickets': [2, 1],
        'Marketing_Engagement': ['High', 'Low'],
        'Customer_Feedback': ['Positive', 'Neutral'],
        'Risk_Appetite': ['Medium', 'Low'],
        'Renewal_Behavior': ['On Time', 'Late']
    }
    template_df = pd.DataFrame(template_data)
    return template_df

# Download template
def download_template():
    template_df = generate_template()
    buffer = io.BytesIO()
    template_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer

# Function for customer segmentation
def segment_customers(df):
    # Preprocessing: Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Gender', 'Income_Level', 'Occupation', 'Marital_Status', 'Education_Level', 
                        'Location', 'Policy_Type', 'Claim_History', 'Payment_Preferences', 
                        'Marketing_Engagement', 'Customer_Feedback', 'Risk_Appetite', 'Renewal_Behavior']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Selecting features
    features = ['Age', 'Gender', 'Income_Level', 'Occupation', 'Marital_Status', 'Education_Level', 
                'Location', 'Coverage_Amount', 'Premium_Amount', 'Policy_Tenure_Years', 
                'Claim_History', 'Payment_Preferences', 'Customer_Support_Tickets', 
                'Marketing_Engagement', 'Customer_Feedback', 'Risk_Appetite', 'Renewal_Behavior']

    X = df[features]

    # Scaling the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Analyzing clusters
    cluster_analysis = df.groupby('Cluster').mean()
    product_potential = df.groupby(['Cluster', 'Policy_Type']).size().unstack().fillna(0)

    return df, cluster_analysis, product_potential

# Streamlit App Interface
st.title("Customer Segmentation App")

st.write("### Step 1: Download CSV Template")
st.write("Download the CSV template and use it to prepare your customer data for segmentation.")
template_buffer = download_template()
st.download_button(
    label="Download CSV Template",
    data=template_buffer,
    file_name="customer_segmentation_template.csv",
    mime="text/csv"
)

st.write("### Step 2: Upload Your Customer Data")
uploaded_file = st.file_uploader("Upload a CSV file prepared using the template:", type="csv")

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    
    # Debugging: Show uploaded data
    st.write("Uploaded Data Preview:")
    st.write(df_uploaded.head())

    try:
        # Perform segmentation
        segmented_df, cluster_analysis, product_potential = segment_customers(df_uploaded)

        st.write("### Step 3: Segmentation Results and Insights")

        # Show cluster analysis
        st.write("**Cluster Analysis (Average Characteristics):**")
        st.write(cluster_analysis)

        # Show product potential
        st.write("**Product Potential by Cluster:**")
        st.write(product_potential)

        # Visualize the clusters
        st.write("**Cluster Distribution Visualization:**")
        st.bar_chart(segmented_df['Cluster'].value_counts())
    except Exception as e:
        st.error(f"An error occurred during segmentation: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
