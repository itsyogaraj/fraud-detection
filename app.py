import streamlit as st
import pandas as pd
import numpy as np
import joblib
import holidays


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🔍",
    layout="wide"
)


# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():
    model_data = joblib.load("fraud_detection_model.pkl")
    return model_data["model"], model_data["features"]

model, model_features = load_model()


# -----------------------------
# Title
# -----------------------------

st.title("🔍 Fraud Detection System")

st.write(
    "Enter application details below to predict whether "
    "the application may be fraudulent."
)


# -----------------------------
# Input
# -----------------------------

application_date = st.date_input(
    "Application Date"
)

application_time = st.time_input(
    "Application Time"
)


# -----------------------------
# Create datetime
# -----------------------------

application_datetime = pd.Timestamp(
    f"{application_date} {application_time}"
)


# -----------------------------
# Date Features
# -----------------------------

input_data = pd.DataFrame({
    "year": [application_datetime.year],
    "month": [application_datetime.month],
    "day": [application_datetime.day],
    "day_name": [application_datetime.day_name()],
    "is_weekend": [
        application_datetime.dayofweek >= 5
    ],
    "day_type": [
        "Weekend"
        if application_datetime.dayofweek >= 5
        else "Weekday"
    ],
    "hour": [application_datetime.hour],
    "minute": [application_datetime.minute]
})


# -----------------------------
# Time Bucket
# -----------------------------

hour = application_datetime.hour

if 5 <= hour < 12:
    time_bucket = "Morning"
elif 12 <= hour < 17:
    time_bucket = "Afternoon"
elif 17 <= hour < 21:
    time_bucket = "Evening"
else:
    time_bucket = "Night"

input_data["time_bucket"] = time_bucket


# -----------------------------
# India Holiday
# -----------------------------

india_holidays = holidays.India()

input_data["Is_Holiday"] = int(
    application_datetime.date() in india_holidays
)


# -----------------------------
# Match Training Features
# -----------------------------

input_data = pd.get_dummies(
    input_data
)

input_data = input_data.reindex(
    columns=model_features,
    fill_value=0
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Fraud"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Potential Fraud Detected")
    else:
        st.success("✅ Application Appears Non-Fraudulent")

    st.metric(
        "Fraud Probability",
        f"{probability:.2%}"
    )