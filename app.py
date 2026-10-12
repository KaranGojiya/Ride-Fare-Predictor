import streamlit as st
import pandas as pd
import joblib


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
    return joblib.load("ride_fare_xgboost_model.pkl")


model = load_model()


# ==============================
# Header
# ==============================
st.title("Smart Ride Fare Predictor")
st.write(
    "Predict the final ride fare using city, ride details, demand conditions, "
    "surge, traffic delay, and ride priority."
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
# Input Section
# ==============================
st.subheader("Enter Ride Details")

col1, col2, col3 = st.columns(3)

with col1:
    city = st.selectbox(
        "City",
        ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Mumbai"]
    )

    day_of_week = st.selectbox(
        "Day of Week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday"
        }[x]
    )

    ride_distance = st.number_input(
        "Ride Distance KM",
        min_value=0.50,
        max_value=19.99,
        value=8.50,
        step=0.10
    )

    ride_type = st.selectbox(
        "Ride Type",
        ["Economy", "Luxury", "Premium"]
    )

    weather = st.selectbox(
        "Weather",
        ["Clear", "Cloudy", "Rainy", "Stormy"]
    )

    event = st.selectbox(
        "Event",
        ["Concert", "Festival", "IPL Match", "Political Rally"]
    )


with col2:
    payment_type = st.selectbox(
        "Payment Type",
        ["Card", "Cash", "UPI"]
    )

    available_drivers = st.number_input(
        "Available Drivers",
        min_value=0.200075,
        max_value=0.999884,
        value=0.650000,
        step=0.01,
        format="%.6f"
    )

    demand_level = st.number_input(
        "Demand Level",
        min_value=0.400019,
        max_value=0.999996,
        value=0.700000,
        step=0.01,
        format="%.6f"
    )

    surge_multiplier = st.number_input(
        "Surge Multiplier",
        min_value=1.00,
        max_value=2.20,
        value=1.20,
        step=0.01
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


with col3:
    time_of_day = st.selectbox(
        "Time of Day",
        ["Afternoon", "Evening Peak", "Morning Peak", "Night"]
    )

    demand_score = st.selectbox(
        "Demand Score",
        [50, 65, 70, 80, 85, 95, 100]
    )

    driver_availability = st.number_input(
        "Driver Availability",
        min_value=30,
        max_value=100,
        value=70,
        step=1
    )

    traffic_delay = st.selectbox(
        "Traffic Delay",
        [0, 10, 15, 25]
    )

    ride_priority = st.number_input(
        "Ride Priority",
        min_value=83.05,
        max_value=99.99,
        value=90.00,
        step=0.10
    )


# ==============================
# Input DataFrame
# ==============================
input_data = pd.DataFrame({
    "City": [city],
    "Day_of_Week": [day_of_week],
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
    "Time_of_Day": [time_of_day],
    "Demand_Score": [demand_score],
    "Driver_Availability": [driver_availability],
    "Traffic_Delay": [traffic_delay],
    "Ride_Priority": [ride_priority]
})


# ==============================
# Prediction
# ==============================
st.divider()
st.subheader("Prediction")

if st.button("Predict Final Fare", use_container_width=True):
    try:
        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Final Fare: ₹{prediction:,.2f}")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Distance", f"{ride_distance:.2f} km")

        with c2:
            st.metric("Surge", f"{surge_multiplier:.2f}x")

        with c3:
            st.metric("Traffic Delay", f"{traffic_delay} min")

        with c4:
            st.metric("Fare per KM", f"₹{prediction / ride_distance:,.2f}")

        st.markdown("### Ride Summary")
        st.dataframe(input_data, use_container_width=True)

        st.markdown("### Fare Insights")

        insights = []

        if ride_type in ["Premium", "Luxury"]:
            insights.append("Selected ride type has a higher pricing category.")

        if surge_multiplier >= 1.5:
            insights.append("Surge multiplier is high, so fare is likely to increase.")

        if demand_score >= 85:
            insights.append("Demand score is high, which can increase the fare.")

        if driver_availability <= 45:
            insights.append("Driver availability is low, which may increase pricing pressure.")

        if traffic_delay >= 15:
            insights.append("Traffic delay is significant and may increase fare.")

        if is_weekend == 1:
            insights.append("Weekend ride patterns may influence fare.")

        if not insights:
            insights.append("Ride conditions look normal based on selected inputs.")

        for insight in insights:
            st.write("-", insight)

    except Exception as e:
        st.error("Prediction failed. Please check that the app columns match the model training columns.")
        st.exception(e)


# ==============================
# Model Performance
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

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Selected Model", "XGBoost Regressor")

with m2:
    st.metric("MAE", "2.35")

with m3:
    st.metric("R² Score", "0.9983")

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
