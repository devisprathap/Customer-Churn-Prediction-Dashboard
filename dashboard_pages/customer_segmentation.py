import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def show():

    st.title(" Customer Segmentation")

    st.write(
        "This page uses K-Means clustering to group customers "
        "based on their tenure, monthly charges, and total charges."
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
    # Scale Data
    # ------------------------------------------

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(features)

    # ------------------------------------------
    # Apply K-Means
    # ------------------------------------------

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df["Customer Segment"] = kmeans.fit_predict(
        scaled_features
    )

    # ------------------------------------------
    # Display Segment Counts
    # ------------------------------------------

    st.subheader(" Customer Segment Summary")

    segment_counts = df["Customer Segment"].value_counts().sort_index()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Segment 0",
            int(segment_counts.get(0, 0))
        )

    with col2:
        st.metric(
            "Segment 1",
            int(segment_counts.get(1, 0))
        )

    with col3:
        st.metric(
            "Segment 2",
            int(segment_counts.get(2, 0))
        )

    # ------------------------------------------
    # Visualize Segments
    # ------------------------------------------

    st.subheader(" Customer Segments")

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        df["tenure"],
        df["MonthlyCharges"],
        c=df["Customer Segment"],
        cmap="viridis",
        alpha=0.6
    )

    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Monthly Charges")
    ax.set_title("Customer Segments by Tenure and Monthly Charges")

    fig.colorbar(
        scatter,
        ax=ax,
        label="Customer Segment"
    )

    st.pyplot(fig)

    # ------------------------------------------
    # Segment Characteristics
    # ------------------------------------------

    st.subheader(" Segment Characteristics")

    summary = df.groupby("Customer Segment")[
        ["tenure", "MonthlyCharges", "TotalCharges"]
    ].mean().round(2)

    st.dataframe(
        summary,
        use_container_width=True
    )

    # ------------------------------------------
    # View Customer Data
    # ------------------------------------------

    st.subheader(" Customer Segment Details")

    st.dataframe(
        df[
            [
                "customerID",
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "Customer Segment"
            ]
        ].head(20),
        use_container_width=True
    )