import streamlit as st
from dashboard_pages import overview,churn_analysis,customer_analysis,revenue_analysis,prediction,model_performance,insights,customer_segmentation,anomaly_detection
from dashboard_pages import sales_trend
from dashboard_pages import time_series_forecasting
from dashboard_pages import arima_forecasting
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="",
    layout="wide"
)

# Sidebar
st.sidebar.title(" Churn Dashboard")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Churn Analysis",
        "Customer Analysis",
        "Revenue Analysis",
        "Prediction",
        "Model Performance",
        "Insights",
        "Customer Segmentation",
        "Anomaly Detection",
        "Sales Trend Analysis",
        "Time Series Forecasting",
        "ARIMA Forecasting"
    ]
)

# Display selected page

if page == "Overview":
    overview.show()

elif page == "Churn Analysis":
    churn_analysis.show()

elif page == "Customer Analysis":
    customer_analysis.show()

elif page == "Revenue Analysis":
    revenue_analysis.show()

elif page == "Prediction":
    prediction.show()

elif page == "Model Performance":
    model_performance.show()

elif page == "Insights":
    insights.show()
elif page == "Customer Segmentation":
    customer_segmentation.show()
    
elif page == "Anomaly Detection":
    anomaly_detection.show()
elif page == "Sales Trend Analysis":
    sales_trend.show()
elif page == "Time Series Forecasting":
    time_series_forecasting.show()
elif page == "ARIMA Forecasting":
    arima_forecasting.show()