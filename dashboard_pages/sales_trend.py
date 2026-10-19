
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def show():

    st.title(" Sales Trend Analysis")

    st.write(
        "Analyze simulated customer charge trends by month, "
        "quarter, and year."
    )

    # Load dataset with simulated dates
    df = pd.read_csv("data/customer_churn_with_dates.csv")

    # Convert columns to correct data types
    df["TransactionDate"] = pd.to_datetime(
        df["TransactionDate"], errors="coerce"
    )

    df["MonthlyCharges"] = pd.to_numeric(
        df["MonthlyCharges"], errors="coerce"
    )

    df = df.dropna(
        subset=["TransactionDate", "MonthlyCharges"]
    ).copy()

    # Calculate monthly revenue proxy
    df["YearMonth"] = df["TransactionDate"].dt.to_period("M")
    df["Quarter"] = df["TransactionDate"].dt.to_period("Q")
    df["Year"] = df["TransactionDate"].dt.year

    st.info(
        "The dates are simulated for demonstration. "
        "These charts do not represent actual historical sales."
    )

    # Revenue summary
    st.subheader(" Customer Charge Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Customer Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Sum of Monthly Charges",
        f"₹{df['MonthlyCharges'].sum():,.2f}"
    )

    col3.metric(
        "Average Monthly Charge",
        f"₹{df['MonthlyCharges'].mean():,.2f}"
    )

    # Monthly trend
    st.subheader(" Monthly Trend")

    monthly_sales = (
        df.groupby("YearMonth")["MonthlyCharges"]
        .sum()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_sales.index = monthly_sales.index.astype(str)
    monthly_sales.plot(kind="line", marker="o", ax=ax)

    ax.set_xlabel("Month")
    ax.set_ylabel("Sum of Customer Monthly Charges (₹)")
    ax.set_title("Monthly Customer Charge Trend")
    ax.tick_params(axis="x", rotation=60)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Quarterly trend
    st.subheader(" Quarterly Trend")

    quarterly_sales = (
        df.groupby("Quarter")["MonthlyCharges"]
        .sum()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    quarterly_sales.index = quarterly_sales.index.astype(str)
    quarterly_sales.plot(kind="bar", ax=ax)

    ax.set_xlabel("Quarter")
    ax.set_ylabel("Sum of Customer Monthly Charges (₹)")
    ax.set_title("Quarterly Customer Charge Trend")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Yearly trend
    st.subheader(" Yearly Trend")

    yearly_sales = (
        df.groupby("Year")["MonthlyCharges"]
        .sum()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    yearly_sales.plot(kind="bar", ax=ax)

    ax.set_xlabel("Year")
    ax.set_ylabel("Sum of Customer Monthly Charges (₹)")
    ax.set_title("Yearly Customer Charge Trend")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Show summarized data
    with st.expander("View Monthly Summary Data"):
        st.dataframe(
            monthly_sales.rename("Total Monthly Charges"),
            use_container_width=True
        )

    with st.expander("View Quarterly Summary Data"):
        st.dataframe(
            quarterly_sales.rename("Total Monthly Charges"),
            use_container_width=True
        )

    with st.expander("View Yearly Summary Data"):
        st.dataframe(
            yearly_sales.rename("Total Monthly Charges"),
            use_container_width=True
        )

