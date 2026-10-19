import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import joblib

def show():

    st.title(" Model Performance")

    st.write(
        "This page compares the performance of different "
        "machine learning models used for customer churn prediction."
    )

    # Model performance results
    data = {
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting"
        ],
        "Accuracy": [
            0.8013,
            0.7864,
            0.7984
        ],
        "Precision": [
            0.6460,
            0.6237,
            0.6531
        ],
        "Recall": [
            0.5561,
            0.4920,
            0.5134
        ],
        "ROC-AUC": [
            0.8426,
            0.8251,
            0.8425
        ]
    }

    df = pd.DataFrame(data)

    # Display table
    st.subheader("Model Comparison")

    st.dataframe(
        df,
        use_container_width=True
    )

    # Best model
    st.subheader(" Selected Model")

    st.success(
        "Logistic Regression was selected because it provides "
        "the best overall performance, especially in Accuracy "
        "and Recall."
    )
    # ------------------------------------------
    # Confusion Matrix
    # ------------------------------------------

    st.subheader(" Confusion Matrix - Logistic Regression")

    # Load dataset
    df = pd.read_csv("data/customer_churn.csv")

    # Clean TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # Remove customer ID
    df = df.drop(columns=["customerID"])

    # Encode target
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Encode categorical columns
    df = pd.get_dummies(
        df,
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

    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Create test data
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Load trained model
    model = joblib.load("models/churn_model.pkl")

    # Match training columns
    X_test = X_test.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Predict
    y_pred = model.predict(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Display heatmap
    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed", "Churned"],
        yticklabels=["Stayed", "Churned"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Customer Churn Confusion Matrix")

    st.pyplot(fig)
    # ------------------------------------------
    # Classification Report
    # ------------------------------------------

    st.subheader(" Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Stayed", "Churned"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )