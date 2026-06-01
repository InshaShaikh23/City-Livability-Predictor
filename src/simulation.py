import streamlit as st
import pandas as pd

def run_simulation(latest, model, feature_cols, city_name):
    st.subheader("🎚 Policy Simulation (What-If Analysis)")
    st.write("Adjust key indicators to see how livability changes:")

    sim_data = latest.copy()

    # Sliders
    sim_data["Literacy_Rate_%"] = st.slider(
        "Literacy Rate (%)",
        50.0, 100.0,
        float(latest["Literacy_Rate_%"])
    )

    sim_data["Internet_Penetration_%"] = st.slider(
        "Internet Penetration (%)",
        40.0, 100.0,
        float(latest["Internet_Penetration_%"])
    )

    sim_data["Average_AQI"] = st.slider(
        "Average AQI (Lower is better)",
        50.0, 300.0,
        float(latest["Average_AQI"])
    )

    sim_data["Crime_Rate_per_100k"] = st.slider(
        "Crime Rate per 100k (Lower is better)",
        0.0, 1000.0,
        float(latest["Crime_Rate_per_100k"])
    )

    # Prepare input for model
    sim_X = pd.DataFrame([sim_data])[feature_cols]

    # Predict
    sim_score = model.predict(sim_X)[0]

    st.success(f"🏙 Simulated Livability Score for {city_name} : {sim_score:.2f}")

    return sim_score
