import streamlit as st
import pandas as pd
import joblib


def show():

    st.title(" Customer Churn Prediction")

    st.write(
        "Enter customer details below to predict "
        "whether the customer is likely to churn."
    )

    # Load trained model
    model = joblib.load("models/churn_model.pkl")

    # ------------------------------------------
    # Customer Information
    # ------------------------------------------

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12
        )

    with col2:
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

    st.subheader("Services")

    col1, col2, col3 = st.columns(3)

    with col1:
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

    with col2:
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

    with col3:
        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
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

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0
        )

    # ------------------------------------------
    # Create Input DataFrame
    # ------------------------------------------

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })
    # ------------------------------------------
    # One-Hot Encoding
    # ------------------------------------------

    input_data = pd.get_dummies(
        input_data,
        columns=[
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod"
        ],
        drop_first=True
    )
    # ------------------------------------------
    # Make input columns exactly the same as training
    # ------------------------------------------

    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # ------------------------------------------
    # Predict Churn
    # ------------------------------------------

    if st.button(" Predict Churn"):

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader(" Prediction Result")

        if prediction == 1:
            st.error(" Customer is likely to churn.")
        else:
            st.success(" Customer is likely to stay.")

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )