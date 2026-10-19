import streamlit as st
import pandas as pd


def show():

    st.title(" Churn Insights & Recommendations")

    st.write(
        "This page provides business insights from customer "
        "churn analysis and suggests actions to improve "
        "customer retention."
    )

    # Load data
    df = pd.read_csv("data/customer_churn.csv")

    # ------------------------------------------
    # Contract Analysis
    # ------------------------------------------

    st.subheader(" Contract Insights")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100

    highest_contract = contract_churn["Yes"].idxmax()
    highest_contract_rate = contract_churn["Yes"].max()

    st.write(
        f"**{highest_contract}** customers have the highest "
        f"churn rate at approximately **{highest_contract_rate:.2f}%**."
    )

    # ------------------------------------------
    # Internet Service Analysis
    # ------------------------------------------

    st.subheader(" Internet Service Insights")

    internet_churn = pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ) * 100

    highest_internet = internet_churn["Yes"].idxmax()
    highest_internet_rate = internet_churn["Yes"].max()

    st.write(
        f"Customers using **{highest_internet}** have the highest "
        f"churn rate at approximately **{highest_internet_rate:.2f}%**."
    )

    # ------------------------------------------
    # Monthly Charges Analysis
    # ------------------------------------------

    st.subheader("Monthly Charges Insights")

    df["MonthlyCharges"] = pd.to_numeric(
        df["MonthlyCharges"],
        errors="coerce"
    )

    avg_charges = df.groupby("Churn")["MonthlyCharges"].mean()

    stayed_charges = avg_charges["No"]
    churned_charges = avg_charges["Yes"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Average Charges - Stayed",
            f"₹{stayed_charges:.2f}"
        )

    with col2:
        st.metric(
            "Average Charges - Churned",
            f"₹{churned_charges:.2f}"
        )

    # ------------------------------------------
    # Recommendations
    # ------------------------------------------

    st.subheader("Recommendations")

    st.info(
        "1. Encourage month-to-month customers to move to "
        "longer-term contracts."
    )

    st.info(
        "2. Provide special retention offers to customers "
        "with higher monthly charges."
    )

    st.info(
        "3. Monitor customers using high-churn service types "
        "and provide targeted support."
    )

    st.info(
        "4. Use the churn prediction model to identify "
        "high-risk customers early."
    )