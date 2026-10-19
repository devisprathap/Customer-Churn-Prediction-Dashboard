import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def show():

    st.title(" Churn Analysis")

    # Load data
    df = pd.read_csv("data/customer_churn.csv")

    # Churn count
    churn_counts = df["Churn"].value_counts()

    st.subheader("Customer Churn Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Customers Stayed", churn_counts["No"])
        st.metric("Customers Churned", churn_counts["Yes"])

    with col2:
        fig, ax = plt.subplots()

        ax.bar(
            ["Stayed", "Churned"],
            [churn_counts["No"], churn_counts["Yes"]]
        )

        ax.set_xlabel("Customer Status")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Customer Churn Distribution")

        st.pyplot(fig)

    # Churn percentage
    churn_rate = (churn_counts["Yes"] / len(df)) * 100

    st.subheader("Churn Rate")

    st.metric(
        "Overall Churn Rate",
        f"{churn_rate:.2f}%"
    )