
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


def show():
    st.title("ARIMA Time Series Forecasting")

    st.write(
        "ARIMA uses historical time-series patterns "
        "to forecast future values."
    )

    st.info(
        "This demonstration uses simulated dates. "
        "The forecasts are not actual business revenue predictions."
    )

    # Step 1: Load dataset
    try:
        df = pd.read_csv("data/customer_churn_with_dates.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Please check the data folder.")
        return

    # Step 2: Prepare data
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

    # Step 3: Aggregate charges by month
    monthly_data = (
        df.set_index("TransactionDate")["MonthlyCharges"]
        .resample("MS")
        .sum()
        .asfreq("MS", fill_value=0)
    )

    if len(monthly_data) < 12:
        st.error("At least 12 months of data are needed for this demo.")
        return

    st.subheader("Historical Monthly Charges")

    fig, ax = plt.subplots()
    ax.plot(monthly_data.index, monthly_data.values, marker="o")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Monthly Charges")
    ax.set_title("Monthly Charges Over Time")
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

    # Step 4: Train ARIMA model
    st.subheader("ARIMA Model Forecast")

    try:
        model = ARIMA(
            monthly_data,
            order=(1, 1, 1)
        )

        fitted_model = model.fit()

        # Step 5: Forecast the next 6 months
        forecast_result = fitted_model.get_forecast(steps=6)

        forecast_values = forecast_result.predicted_mean
        confidence_intervals = forecast_result.conf_int()

        # Step 6: Display forecast chart
        fig, ax = plt.subplots()

        ax.plot(
            monthly_data.index,
            monthly_data.values,
            label="Historical Charges"
        )

        ax.plot(
            forecast_values.index,
            forecast_values.values,
            marker="o",
            label="ARIMA Forecast"
        )

        ax.fill_between(
            forecast_values.index,
            confidence_intervals.iloc[:, 0],
            confidence_intervals.iloc[:, 1],
            alpha=0.2,
            label="Confidence Interval"
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Total Monthly Charges")
        ax.set_title("ARIMA: Six-Month Forecast")
        ax.legend()
        fig.autofmt_xdate()

        st.pyplot(fig)
        plt.close(fig)

        # Step 7: Display forecast table
        st.subheader("Forecast Values")

        forecast_table = pd.DataFrame({
            "Month": forecast_values.index,
            "Predicted Charges": forecast_values.values,
            "Lower Estimate": confidence_intervals.iloc[:, 0].values,
            "Upper Estimate": confidence_intervals.iloc[:, 1].values
        })

        st.dataframe(
            forecast_table.round(2),
            use_container_width=True
        )

        # Step 8: Display model summary
        with st.expander("View ARIMA Model Summary"):
            st.text(fitted_model.summary().as_text())

    except Exception as e:
        st.error(f"ARIMA forecasting failed: {e}")
