import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load saved files
# -----------------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
ohe = joblib.load("onehot_encoder.pkl")


# -----------------------------
# Page
# -----------------------------
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer will churn.")


# -----------------------------
# User Inputs
# -----------------------------

gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])

partner = st.selectbox("Partner", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure Months", min_value=0, max_value=100, value=1)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=50.0
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No phone service", "No", "Yes"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Churn"):

    # Raw input
    input_data = pd.DataFrame({
        "Gender": [gender],
        "Senior Citizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "Tenure Months": [tenure],
        "Phone Service": [phone_service],
        "Paperless Billing": [paperless_billing],
        "Monthly Charges": [monthly_charges],
        "Total Charges": [total_charges],
        "Multiple Lines": [multiple_lines],
        "Internet Service": [internet_service],
        "Online Security": [online_security],
        "Online Backup": [online_backup],
        "Device Protection": [device_protection],
        "Tech Support": [tech_support],
        "Streaming TV": [streaming_tv],
        "Streaming Movies": [streaming_movies],
        "Contract": [contract],
        "Payment Method": [payment_method]
    })


    # -----------------------------
    # Label Encoding
    # -----------------------------

    label_columns = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Phone Service",
        "Paperless Billing"
    ]

    for col in label_columns:
        input_data[col] = label_encoders[col].transform(input_data[col])


    # -----------------------------
    # One Hot Encoding
    # -----------------------------

    multi_columns = [
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Payment Method"
    ]

    encoded = ohe.transform(input_data[multi_columns])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(multi_columns)
    )


    # Remove original categorical columns
    input_data = input_data.drop(columns=multi_columns)


    # Combine
    final_data = pd.concat(
        [input_data, encoded_df],
        axis=1
    )
    

    # -----------------------------
    # Exact model column order
    # -----------------------------

    columns = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure Months",
        "Phone Service",
        "Paperless Billing",
        "Monthly Charges",
        "Total Charges",
        "Multiple Lines_No phone service",
        "Multiple Lines_Yes",
        "Internet Service_Fiber optic",
        "Internet Service_No",
        "Online Security_No internet service",
        "Online Security_Yes",
        "Online Backup_No internet service",
        "Online Backup_Yes",
        "Device Protection_No internet service",
        "Device Protection_Yes",
        "Tech Support_No internet service",
        "Tech Support_Yes",
        "Streaming TV_No internet service",
        "Streaming TV_Yes",
        "Streaming Movies_No internet service",
        "Streaming Movies_Yes",
        "Contract_One year",
        "Contract_Two year",
        "Payment Method_Credit card (automatic)",
        "Payment Method_Electronic check",
        "Payment Method_Mailed check"
    ]

    final_data = final_data[columns]


    # -----------------------------
    # Scaling
    # -----------------------------

    final_data = scaler.transform(final_data)


    # -----------------------------
    # Prediction
    # -----------------------------

prediction = model.predict(final_data)[0]
probability = model.predict_proba(final_data)[0][1]

st.write(f"Churn Probability: {probability:.2%}")

if probability >= 0.70:
    st.error("🔴 High Risk — Customer is likely to churn.")
elif probability >= 0.40:
    st.warning("🟡 Medium Risk — Customer may churn.")
else:
    st.success("🟢 Low Risk — Customer is unlikely to churn.")