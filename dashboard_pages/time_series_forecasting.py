
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


def show():
    st.title("📈 Time Series Forecasting using Prophet")

    st.write(
        "This page demonstrates time series forecasting "
        "using the Prophet machine learning library."
    )

    # Load dataset
    try:
        df = pd.read_csv("data/customer_churn_with_dates.csv")
    except FileNotFoundError:
        st.error(
            "Dataset not found. Please check "
            "data/customer_churn_with_dates.csv"
        )
        return

    # Prepare data
    df["TransactionDate"] = pd.to_datetime(
        df["TransactionDate"], errors="coerce"
    )

    df["MonthlyCharges"] = pd.to_numeric(
        df["MonthlyCharges"], errors="coerce"
    )

    df = df.dropna(
        subset=["TransactionDate", "MonthlyCharges"]
    )

    if df.empty:
        st.error("No valid date and charge data available.")
        return

    # Aggregate charges by month
    monthly_data = (
        df.set_index("TransactionDate")["MonthlyCharges"]
        .resample("MS")
        .sum()
        .reset_index()
    )

    # Prophet requires columns named ds and y
    prophet_data = monthly_data.rename(
        columns={
            "TransactionDate": "ds",
            "MonthlyCharges": "y"
        }
    )

    if len(prophet_data) < 3:
        st.error("At least 3 monthly observations are required.")
        return

    # Display historical data
    st.subheader("Monthly Charges")

    fig, ax = plt.subplots()
    ax.plot(
        prophet_data["ds"],
        prophet_data["y"],
        marker="o"
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Monthly Charges")
    ax.set_title("Monthly Charges Over Time")
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

    # Train Prophet model
    st.subheader("Future Forecast")

    model = Prophet()
    model.fit(prophet_data)

    # Forecast the next 6 months
    future = model.make_future_dataframe(
        periods=6,
        freq="MS"
    )

    forecast = model.predict(future)

    # Display forecast chart
    forecast_fig = model.plot(forecast)
    st.pyplot(forecast_fig)
    plt.close(forecast_fig)

    # Display forecast components
    st.subheader("Forecast Components")

    component_fig = model.plot_components(forecast)
    st.pyplot(component_fig)
    plt.close(component_fig)

    # Display forecast table
    st.subheader("Forecast Values")

    future_forecast = forecast[
        forecast["ds"] > prophet_data["ds"].max()
    ][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    future_forecast.columns = [
        "Date",
        "Predicted Charges",
        "Lower Estimate",
        "Upper Estimate"
    ]

    st.dataframe(
        future_forecast.round(2),
        use_container_width=True
    )

    st.info(
        "Note: The dates in this demonstration dataset are simulated. "
        "These forecasts demonstrate Prophet's functionality and should "
        "not be treated as predictions of actual business revenue."
    )

