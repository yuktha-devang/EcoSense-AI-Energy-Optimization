
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
data = pd.read_csv("ecosense_cleaned_data.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"])

st.success("Energy dataset loaded successfully!")

# Building selection
building = st.selectbox(
    "Select Building",
    ["building_1", "building_2"]
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
