import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def show():

    st.title(" Customer Analysis")

    # Load data
    df = pd.read_csv("data/customer_churn.csv")

    # ------------------------------------------
    # Churn by Contract
    # ------------------------------------------

    st.subheader("Churn by Contract Type")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )

    st.bar_chart(contract_churn)

    # ------------------------------------------
    # Churn by Internet Service
    # ------------------------------------------

    st.subheader("Churn by Internet Service")

    internet_churn = pd.crosstab(
        df["InternetService"],
        df["Churn"]
    )

    st.bar_chart(internet_churn)

    # ------------------------------------------
    # Churn by Gender
    # ------------------------------------------

    st.subheader("Churn by Gender")

    gender_churn = pd.crosstab(
        df["gender"],
        df["Churn"]
    )

    st.bar_chart(gender_churn)

    # ------------------------------------------
    # Average Monthly Charges
    # ------------------------------------------

    st.subheader("Average Monthly Charges")

    avg_charges = df.groupby("Churn")["MonthlyCharges"].mean()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Average Charges - Stayed",
            f"₹{avg_charges['No']:.2f}"
        )

    with col2:
        st.metric(
            "Average Charges - Churned",
            f"₹{avg_charges['Yes']:.2f}"
        )