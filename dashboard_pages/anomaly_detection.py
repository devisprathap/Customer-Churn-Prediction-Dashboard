import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


def show():

    st.title(" Customer Anomaly Detection")

    st.write(
        "This page uses DBSCAN to identify unusual customer "
        "patterns based on tenure, monthly charges, and total charges."
    )

    # ------------------------------------------
    # Load Dataset
    # ------------------------------------------

    df = pd.read_csv("data/customer_churn.csv")

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove rows with missing values in selected features
    df = df.dropna(
        subset=["tenure", "MonthlyCharges", "TotalCharges"]
    ).copy()

    # ------------------------------------------
    # Select Features
    # ------------------------------------------

    features = df[
        ["tenure", "MonthlyCharges", "TotalCharges"]
    ]

    # ------------------------------------------
    # Scale Features
    # ------------------------------------------

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(features)

    # ------------------------------------------
    # Apply DBSCAN
    # ------------------------------------------

    dbscan = DBSCAN(
        eps=0.8,
        min_samples=10
    )

    df["Cluster"] = dbscan.fit_predict(scaled_features)

    # DBSCAN labels noise points as -1
    anomalies = df[df["Cluster"] == -1]

    normal_customers = df[df["Cluster"] != -1]

    # ------------------------------------------
    # Display Summary
    # ------------------------------------------

    st.subheader("Anomaly Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Customers Analyzed",
            len(df)
        )

    with col2:
        st.metric(
            "Potential Anomalies",
            len(anomalies)
        )

    with col3:
        anomaly_rate = (
            len(anomalies) / len(df) * 100
            if len(df) > 0 else 0
        )

        st.metric(
            "Anomaly Percentage",
            f"{anomaly_rate:.2f}%"
        )

    # ------------------------------------------
    # Visualize Results
    # ------------------------------------------

    st.subheader(" Customer Anomaly Visualization")

    fig, ax = plt.subplots()

    ax.scatter(
        normal_customers["tenure"],
        normal_customers["MonthlyCharges"],
        label="Normal Patterns",
        alpha=0.6
    )

    ax.scatter(
        anomalies["tenure"],
        anomalies["MonthlyCharges"],
        label="Potential Anomalies",
        marker="x"
    )

    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Monthly Charges")
    ax.set_title("DBSCAN Customer Anomaly Detection")
    ax.legend()

    st.pyplot(fig)

    # ------------------------------------------
    # Display Potential Anomalies
    # ------------------------------------------

    st.subheader(" Potential Anomalous Customers")

    if len(anomalies) > 0:

        st.dataframe(
            anomalies[
                [
                    "customerID",
                    "tenure",
                    "MonthlyCharges",
                    "TotalCharges",
                    "Cluster"
                ]
            ].head(50),
            use_container_width=True
        )

    else:
        st.info(
            "No anomalies were identified with the current DBSCAN settings."
        )

    st.caption(
        "Note: DBSCAN identifies unusual patterns in the selected "
        "features. An anomaly is not necessarily a bad customer, "
        "fraudulent transaction, or churn risk."
    )