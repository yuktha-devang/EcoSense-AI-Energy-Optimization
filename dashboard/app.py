
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import joblib

st.set_page_config(
    page_title="EcoSense AI",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 EcoSense AI")
st.subheader("Intelligent Energy Consumption Optimization System")

st.write(
    "Monitor building energy readings, "
    "view consumption patterns and receive AI-based recommendations."
)

# Load cleaned dataset
data = pd.read_csv("data/ecosense_cleaned_data.csv")
with gzip.open("models/building_1_model.joblib.gz", "rb") as f:
    model_1 = joblib.load(f)

with gzip.open("models/building_2_model.joblib.gz", "rb") as f:
    model_2 = joblib.load(f)

st.success("AI forecasting models loaded successfully!")

data["timestamp"] = pd.to_datetime(data["timestamp"])

st.success("Energy dataset loaded successfully!")

# Building selection
building = st.selectbox(
    "Select Building",
    ["building_1", "building_2"]
)
st.subheader("🔮 Next-Hour Energy Forecast")

if building == "building_1":
    selected_model = model_1
else:
    selected_model = model_2

data["timestamp"] = pd.to_datetime(data["timestamp"])

latest_time = data["timestamp"].iloc[-1]

prediction_input = pd.DataFrame([{
    "hour": latest_time.hour,
    "day_of_week": latest_time.dayofweek,
    "month": latest_time.month,
    "lag_1": data[building].iloc[-1],
    "lag_24": data[building].iloc[-24],
    "rolling_mean_24": data[building].tail(24).mean()
}])

prediction = selected_model.predict(prediction_input)[0]

st.metric(
    "Predicted Next-Hour Reading",
    round(prediction, 2)
)
average = data[building].mean()
latest = data[building].iloc[-1]

# Display metrics
col1, col2 = st.columns(2)

col1.metric("Latest Reading", round(latest, 2))
col2.metric("Average Reading", round(average, 2))

# Energy consumption graph
st.subheader("Energy Consumption Trend")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(data["timestamp"], data[building])
ax.set_xlabel("Time")
ax.set_ylabel("Energy Reading")
ax.grid(True)

st.pyplot(fig)

# Recommendation
st.subheader("AI Optimization Recommendation")

if latest > average * 1.20:
    st.warning(
        "High consumption detected. "
        "Check HVAC, lighting and unused equipment."
    )
elif latest > average:
    st.info(
        "Slightly high consumption. "
        "Review operating hours and equipment usage."
    )
else:
    st.success(
        "Consumption is within the normal range. "
        "Continue monitoring."
    )
    st.subheader("🚨 Unusual Energy Consumption Detection")

if len(data) >= 25:

    recent_values = data[building].iloc[-25:-1]
    latest_value = data[building].iloc[-1]

    baseline_mean = recent_values.mean()
    baseline_std = recent_values.std()

    threshold = baseline_mean + (2 * baseline_std)

    st.write("Comparing the latest reading with the previous 24 hours.")

    col1, col2 = st.columns(2)

    col1.metric("24-Hour Average", round(baseline_mean, 2))
    col2.metric("Detection Threshold", round(threshold, 2))

    if latest_value > threshold:
        st.warning(
            "Unusual high consumption detected! "
            "Check HVAC, lighting and unused equipment."
        )
    else:
        st.success(
            "No unusual high consumption detected."
        )

else:
    st.info("Not enough historical readings for detection.")
    st.subheader("📥 Download Energy Analysis Report")

# Prepare report data
report_data = pd.DataFrame({
    "Timestamp": data["timestamp"],
    "Building": building,
    "Energy_Reading": data[building]
})

# Add AI forecast information
report_data["Latest_Reading"] = round(latest, 2)
report_data["Average_Reading"] = round(average, 2)
report_data["Predicted_Next_Hour"] = round(prediction, 2)

# Add anomaly detection result
if len(data) >= 25:
    detection_status = (
        "Unusual High Consumption"
        if latest_value > threshold
        else "No Unusual Consumption"
    )
else:
    detection_status = "Insufficient Historical Data"

report_data["Detection_Result"] = detection_status

# Convert report to CSV
csv_report = report_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Energy Report",
    data=csv_report,
    file_name=f"EcoSense_{building}_Report.csv",
    mime="text/csv"
)
