import streamlit as st


# def show():

#     st.title(" Customer Churn Dashboard")

#     st.write(
#         """
#         This dashboard analyzes customer behavior,
#         predicts customer churn, and provides
#         insights for customer retention.
#         """
#     )

#     st.subheader("Project Overview")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.metric("Total Customers", "7,043")

#     with col2:
#         st.metric("Churned Customers", "1,869")

#     with col3:
#         st.metric("Churn Rate", "26.5%")
import streamlit as st
import pandas as pd


def show():

    st.title(" Customer Churn Dashboard")

    st.write(
        "This dashboard analyzes customer behavior, "
        "predicts customer churn, and provides "
        "insights for customer retention."
    )

    # ------------------------------------------
    # Load Data
    # ------------------------------------------

    df = pd.read_csv("data/customer_churn.csv")

    # ------------------------------------------
    # Calculate KPIs
    # ------------------------------------------

    total_customers = len(df)

    churned_customers = (df["Churn"] == "Yes").sum()

    stayed_customers = (df["Churn"] == "No").sum()

    churn_rate = (
        churned_customers / total_customers
    ) * 100

    # ------------------------------------------
    # KPI Cards
    # ------------------------------------------

    st.subheader(" Customer Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            " Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            " Churned Customers",
            f"{churned_customers:,}"
        )

    with col3:
        st.metric(
            " Customers Stayed",
            f"{stayed_customers:,}"
        )

    with col4:
        st.metric(
            " Churn Rate",
            f"{churn_rate:.2f}%"
        )

    # ------------------------------------------
    # Project Description
    # ------------------------------------------

    st.subheader(" Project Overview")

    st.write(
        """
        The Customer Churn Prediction project uses machine
        learning and data analytics to identify customers
        who are likely to leave the service.

        The dashboard provides customer analysis, churn
        analysis, revenue insights, model performance,
        churn prediction, and business recommendations.
        """
    )