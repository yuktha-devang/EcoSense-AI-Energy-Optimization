
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


st.markdown("""
<style>

/* Main dashboard background */
.stApp {
    background: linear-gradient(135deg, #f0fdf4, #ecfeff);
    color: #1f2937 !important;
}

/* Main headings */
h1 {
    color: #14532d !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #166534 !important;
    font-weight: 800 !important;
}

/* Paragraph text */
.stMarkdown p {
    color: #374151 !important;
}

/* Slider labels */
.stSlider label,
.stSlider label p {
    color: #111827 !important;
    font-weight: 700 !important;
}

/* Widget labels */
[data-testid="stWidgetLabel"] p {
    color: #111827 !important;
    font-weight: 700 !important;
}

/* Metric labels - Current Total, Potential Reduction, etc. */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] * {
    color: #111827 !important;
    font-weight: 900 !important;
    opacity: 1 !important;
}

/* Metric numbers */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #14532d !important;
    font-weight: 900 !important;
}

/* Metric delta */
[data-testid="stMetricDelta"] {
    color: #166534 !important;
}

/* Caption text */
[data-testid="stCaptionContainer"] p {
    color: #374151 !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    background-color: #15803d !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 700 !important;
}

/* Button hover */
.stButton button:hover,
.stDownloadButton button:hover {
    background-color: #166534 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)


st.title("🌱 EcoSense AI")
st.markdown("""
### Intelligent Energy Consumption Optimization System

EcoSense AI is an AI-powered energy analysis platform
designed to monitor, predict and optimize building energy consumption.

**Our main objectives:**
- 📊 Analyze energy consumption patterns.
- 🔮 Predict next-hour energy demand using Machine Learning.
- 🚨 Detect unusual energy consumption.
- 💡 Provide energy optimization recommendations.
- 🌱 Estimate potential energy savings.
""")
col1, col2, col3 = st.columns(3)

col1.metric("🏢 Total Buildings", "2")
col2.metric("🤖 AI Models", "2")
col3.metric("🔮 Forecasting", "Next Hour")
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

st.subheader("💰 Energy Savings Calculator")

st.write(
    "Estimate potential savings by reducing building energy consumption."
)

# Select reduction percentage
reduction = st.slider(
    "Select Consumption Reduction (%)",
    min_value=5,
    max_value=15,
    value=10,
    step=5
)

# Calculate potential reduction
total_readings = data[building].dropna().sum()

potential_savings = total_readings * reduction / 100
remaining_readings = total_readings - potential_savings

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Total Readings",
    round(total_readings, 2)
)

col2.metric(
    "Potential Reduction",
    round(potential_savings, 2)
)

col3.metric(
    "Estimated Remaining Readings",
    round(remaining_readings, 2)
)

st.success(
    f"A {reduction}% reduction could save "
    f"{potential_savings:.2f} reading units "
    f"over the selected dataset period."
)

st.caption(
    "This is a theoretical estimate, not measured energy savings. "
    "The dataset's measurement unit must be confirmed before "
    "converting these values into kWh or electricity costs."
)

st.subheader("🏢 Building Energy Comparison")

comparison_fig, comparison_ax = plt.subplots(figsize=(12, 5))

comparison_ax.plot(
    data["timestamp"],
    data["building_1"],
    label="Building 1",
    linewidth=2
)

comparison_ax.plot(
    data["timestamp"],
    data["building_2"],
    label="Building 2",
    linewidth=2
)

comparison_ax.set_xlabel("Date and Time")
comparison_ax.set_ylabel("Energy Reading")
comparison_ax.set_title("Building 1 vs Building 2")
comparison_ax.legend()
comparison_ax.grid(alpha=0.3)

st.pyplot(comparison_fig)
plt.close(comparison_fig)
st.subheader("📊 Building Energy Consumption Comparison")

comparison_data = data[["building_1", "building_2"]].tail(24)

st.line_chart(comparison_data)

st.caption(
    "Comparison of the latest 24 hourly readings "
    "for Building 1 and Building 2."
)
st.subheader("⚙️ How EcoSense AI Works")

st.markdown("""
<style>
.workflow-box {
    background-color: #e8f5e9;
    color: #14532d;
    padding: 15px;
    margin: 8px 0;
    border-radius: 10px;
    text-align: center;
    font-size: 17px;
    font-weight: bold;
    border: 1px solid #86efac;
}
.workflow-arrow {
    text-align: center;
    color: #15803d;
    font-size: 24px;
}
</style>

<div class="workflow-box">📂 Historical Energy Data</div>
<div class="workflow-arrow">⬇️</div>
<div class="workflow-box">🧹 Data Cleaning & Processing</div>
<div class="workflow-arrow">⬇️</div>
<div class="workflow-box">🤖 Random Forest AI Model</div>
<div class="workflow-arrow">⬇️</div>
<div class="workflow-box">🔍 Prediction & Unusual Usage Detection</div>
<div class="workflow-arrow">⬇️</div>
<div class="workflow-box">💡 Recommendations & Energy Savings</div>
""", unsafe_allow_html=True)
st.subheader("🌱 About EcoSense AI")

st.write("""
EcoSense AI is an intelligent energy consumption analysis
and optimization system developed to help buildings
understand and improve their energy usage.

It uses historical energy data and Machine Learning
to predict future consumption, identify unusual usage
and provide useful energy optimization recommendations.
""")

st.markdown("### 🛠️ Technologies Used")

st.markdown("""
- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Regression
- Streamlit
- Matplotlib
- Google Colab
""")

st.markdown("### 🚀 Future Scope")

st.markdown("""
- Integration with IoT energy sensors.
- Real-time energy consumption monitoring.
- Smart alerts for unusual energy usage.
- Advanced AI-based energy optimization.
- Expansion to more buildings and industries.
""")

st.info(
    "EcoSense AI aims to support smarter buildings "
    "through data-driven energy management."
)

