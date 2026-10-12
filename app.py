import streamlit as st
import pandas as pd
import joblib
from datetime import date


# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Smart Ride Fare Predictor",
    page_icon="🚕",
    layout="wide"
)


# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    model = joblib.load("ride_fare_xgboost_model.pkl")
    return model


model = load_model()


# ==============================
# App Header
# ==============================
st.title("Smart Ride Fare Predictor")
st.write(
    "Predict the final ride fare using ride details, demand conditions, "
    "driver availability, traffic, trust scores, and risk factors."
)

st.divider()


# ==============================
# Sidebar
# ==============================
st.sidebar.title("About Project")
st.sidebar.write(
    "This machine learning app predicts ride fares using a trained "
    "XGBoost regression model."
)

st.sidebar.markdown("### Final Model")
st.sidebar.write("XGBoost Regressor")

st.sidebar.markdown("### Target")
st.sidebar.write("Final_Fare")


# ==============================
# User Inputs
# ==============================

st.subheader("Enter Ride Information")

col1, col2, col3 = st.columns(3)

with col1:
    city = st.selectbox(
        "City",
        ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]
    )

    ride_date = st.date_input("Date", value=date.today())

    day_of_week = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=28.61,
        step=0.01
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=77.20,
        step=0.01
    )

    ride_distance = st.number_input(
        "Ride Distance KM",
        min_value=0.1,
        max_value=100.0,
        value=10.0,
        step=0.5
    )

    ride_type = st.selectbox(
        "Ride Type",
        ["Mini", "Sedan", "SUV", "Bike", "Auto", "Premium"]
    )

    weather = st.selectbox(
        "Weather",
        ["Clear", "Cloudy", "Rainy", "Stormy", "Foggy", "Hot"]
    )

    event = st.selectbox(
        "Event",
        ["No Event", "Concert", "Sports", "Festival", "Conference", "Public Holiday"]
    )

    payment_type = st.selectbox(
        "Payment Type",
        ["Cash", "Card", "UPI", "Wallet"]
    )


with col2:
    available_drivers = st.number_input(
        "Available Drivers",
        min_value=0,
        max_value=1000,
        value=120,
        step=1
    )

    demand_level = st.selectbox(
        "Demand Level",
        ["Low", "Medium", "High", "Very High"]
    )

    surge_multiplier = st.number_input(
        "Surge Multiplier",
        min_value=1.0,
        max_value=5.0,
        value=1.2,
        step=0.1
    )

    hour_of_day = st.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=18
    )

    is_weekend = st.selectbox(
        "Is Weekend",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    driver_performance_score = st.number_input(
        "Driver Performance Score",
        min_value=0.0,
        max_value=10.0,
        value=8.0,
        step=0.1
    )

    driver_xp = st.number_input(
        "Driver XP",
        min_value=0,
        max_value=30,
        value=5,
        step=1
    )

    ride_priority = st.selectbox(
        "Ride Priority",
        ["Low", "Normal", "High", "Urgent"]
    )

    fare_acceptance = st.number_input(
        "Fare Acceptance",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01
    )

    cancellation_rate = st.number_input(
        "Cancellation Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01
    )


with col3:
    time_of_day = st.selectbox(
        "Time of Day",
        ["Morning", "Afternoon", "Evening", "Night"]
    )

    demand_score = st.number_input(
        "Demand Score",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        step=1.0
    )

    driver_availability = st.number_input(
        "Driver Availability",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    traffic_delay = st.number_input(
        "Traffic Delay",
        min_value=0.0,
        max_value=120.0,
        value=12.0,
        step=1.0
    )

    cancellation_probability = st.number_input(
        "Cancellation Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01
    )

    driver_trust_score = st.number_input(
        "Driver Trust Score",
        min_value=0.0,
        max_value=10.0,
        value=8.5,
        step=0.1
    )

    rider_trust_score = st.number_input(
        "Rider Trust Score",
        min_value=0.0,
        max_value=10.0,
        value=8.0,
        step=0.1
    )

    leaderboard_rank = st.number_input(
        "Leaderboard Rank",
        min_value=1,
        max_value=10000,
        value=250,
        step=1
    )


# ==============================
# Create Input DataFrame
# ==============================

input_data = pd.DataFrame({
    "City": [city],
    "Date": [str(ride_date)],
    "Day_of_Week": [day_of_week],
    "Latitude": [latitude],
    "Longitude": [longitude],
    "Ride_Distance_KM": [ride_distance],
    "Ride_Type": [ride_type],
    "Weather": [weather],
    "Event": [event],
    "Payment_Type": [payment_type],
    "Available_Drivers": [available_drivers],
    "Demand_Level": [demand_level],
    "Surge_Multiplier": [surge_multiplier],
    "Hour_of_Day": [hour_of_day],
    "Is_Weekend": [is_weekend],
    "Driver_Performance_Score": [driver_performance_score],
    "Driver_XP": [driver_xp],
    "Ride_Priority": [ride_priority],
    "Fare_Acceptance": [fare_acceptance],
    "Cancellation_Rate": [cancellation_rate],
    "Time_of_Day": [time_of_day],
    "Demand_Score": [demand_score],
    "Driver_Availability": [driver_availability],
    "Traffic_Delay": [traffic_delay],
    "Cancellation_Probability": [cancellation_probability],
    "Driver_Trust_Score": [driver_trust_score],
    "Rider_Trust_Score": [rider_trust_score],
    "Leaderboard_Rank": [leaderboard_rank]
})


# ==============================
# Prediction Section
# ==============================

st.divider()

st.subheader("Prediction")

if st.button("Predict Final Fare", use_container_width=True):
    try:
        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Final Fare: ₹{prediction:,.2f}")

        st.markdown("### Ride Summary")

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric("Distance", f"{ride_distance:.1f} km")

        with s2:
            st.metric("Surge", f"{surge_multiplier:.1f}x")

        with s3:
            st.metric("Demand", demand_level)

        with s4:
            fare_per_km = prediction / ride_distance
            st.metric("Fare per KM", f"₹{fare_per_km:,.2f}")

        st.markdown("### Input Preview")
        st.dataframe(input_data, use_container_width=True)

        st.markdown("### Fare Insight")

        insights = []

        if surge_multiplier >= 1.5:
            insights.append("Surge multiplier is high, so fare may increase.")

        if demand_level in ["High", "Very High"]:
            insights.append("Demand level is high, which can increase fare.")

        if traffic_delay >= 20:
            insights.append("Traffic delay is significant and may affect pricing.")

        if available_drivers < 50:
            insights.append("Low driver availability can increase fare.")

        if is_weekend == 1:
            insights.append("Weekend rides may show different pricing patterns.")

        if len(insights) == 0:
            insights.append("Ride conditions look normal based on selected inputs.")

        for item in insights:
            st.write("-", item)

    except Exception as e:
        st.error("Prediction failed. Please check that input columns match the trained model.")
        st.exception(e)


# ==============================
# Model Performance Section
# ==============================

st.divider()

st.subheader("Model Performance")

performance_data = pd.DataFrame({
    "Model": [
        "XGBoost Regressor",
        "Random Forest Regressor",
        "Linear Regression"
    ],
    "MAE": [
        2.345634,
        2.560370,
        15.471232
    ],
    "RMSE": [
        3.851950,
        4.572395,
        22.414083
    ],
    "R2 Score": [
        0.998393,
        0.997736,
        0.945589
    ],
    "Training Time (s)": [
        0.83,
        24.47,
        0.08
    ]
})

st.dataframe(performance_data, use_container_width=True)

best_model_col1, best_model_col2, best_model_col3 = st.columns(3)

with best_model_col1:
    st.metric("Best Model", "XGBoost")

with best_model_col2:
    st.metric("Best MAE", "2.34")

with best_model_col3:
    st.metric("Best R² Score", "0.9983")

st.write(
    "XGBoost Regressor was selected as the final model because it achieved "
    "the lowest MAE, lowest RMSE, highest R² score, and fast training time."
)


# ==============================
# Footer
# ==============================

st.divider()
st.markdown(
    "Developed by **Karan Gojiya** | "
    "[GitHub](https://github.com/KaranGojiya) | "
    "[LinkedIn](https://linkedin.com/in/karan-gojiya)"
)
