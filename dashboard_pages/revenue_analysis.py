import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def show():

    st.title("Revenue / Charges Analysis")

    # Load data
    df = pd.read_csv("data/customer_churn.csv")
    df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    # ------------------------------------------
    # Total Monthly Charges
    # ------------------------------------------

    total_monthly = df["MonthlyCharges"].sum()

    # ------------------------------------------
    # Total Customer Charges
    # ------------------------------------------

    total_charges = df["TotalCharges"].sum()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Monthly Charges",
            f"₹{total_monthly:,.2f}"
        )

    with col2:
        st.metric(
            "Total Charges",
            f"₹{total_charges:,.2f}"
        )

    # ------------------------------------------
    # Average Charges by Churn
    # ------------------------------------------

    st.subheader("Average Monthly Charges by Churn")

    avg_charges = df.groupby("Churn")["MonthlyCharges"].mean()

    fig, ax = plt.subplots()

    ax.bar(
        ["Stayed", "Churned"],
        [
            avg_charges["No"],
            avg_charges["Yes"]
        ]
    )

    ax.set_xlabel("Customer Status")
    ax.set_ylabel("Average Monthly Charges")
    ax.set_title("Average Monthly Charges by Churn")

    st.pyplot(fig)

    # ------------------------------------------
    # Total Charges by Churn
    # ------------------------------------------

    st.subheader("Total Charges by Churn")

    total_by_churn = df.groupby("Churn")["TotalCharges"].sum()

    st.bar_chart(total_by_churn)

    # ------------------------------------------
    # Contract vs Monthly Charges
    # ------------------------------------------

    st.subheader("Contract Type vs Monthly Charges")

    contract_charges = (
        df.groupby("Contract")["MonthlyCharges"]
        .mean()
    )

    st.bar_chart(contract_charges)