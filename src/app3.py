import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

from simulation import run_simulation
from feature_importance import show_feature_importance



# =========================
# Load data & model
# =========================
df = pd.read_csv("../data/made3.csv").dropna()

model = joblib.load("../models/livability_model_v3.pkl")
feature_cols = joblib.load("../models/feature_columns_v3.pkl")

# =========================
# Drop noisy columns
# =========================
drop_cols = [
    "IPC_Crime_Cases",
    "MSME_Registered_Units",
    "Population_Density_per_sqkm"
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# =========================
# Recreate Livability Index
# =========================
positive_cols = [
    'Literacy_Rate_%','Internet_Penetration_%','Electricity_Access_%',
    'Hospitals_per_100k','Beds_per_1000','Toilet_Access_%',
    'Road_Density_km_per_1000sqkm','Higher_Education_Institutions'
]

negative_cols = [
    'Unemployment_Rate_%','Average_AQI','Crime_Rate_per_100k',
    'Student_Teacher_Ratio','PMAY_Average_Housing_Cost_Lakhs',
    'Population_Growth_%'
]

scaler = MinMaxScaler()

df_pos = pd.DataFrame(scaler.fit_transform(df[positive_cols]), columns=positive_cols)
df_neg = pd.DataFrame(scaler.fit_transform(df[negative_cols]), columns=negative_cols)
df_neg = 1 - df_neg

df["Livability_Index"] = (df_pos.mean(axis=1) * 0.6 + df_neg.mean(axis=1) * 0.4) * 100

# =========================
# Predict current livability for map & ranking
# =========================
X_all = df[feature_cols]
df["Predicted_Livability"] = model.predict(X_all)

# =========================
# Classification Function
# =========================
def classify(score):
    if score < 45:
        return "Low"
    elif score < 55:
        return "Medium"
    else:
        return "High"

df["Class"] = df["Predicted_Livability"].apply(classify)

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("📊 Dashboard Menu")

menu = st.sidebar.radio(
    "Go to",
    [
        "Main Dashboard",
        "City Livability Ranking",
        "Feature Importance Analysis",
        "Policy Simulation (What-If)"
    ]
)

st.sidebar.markdown("## 📊 Model Performance")
st.sidebar.write("R² Score: 0.7055")
st.sidebar.write("MAE: 3.19")

# =========================
# MAIN TITLE
# =========================
st.title("🏙️ Indian City Livability Prediction System")

# =========================
# CITY SELECTOR
# =========================
city = st.selectbox("Select City", sorted(df["City"].unique()))
city_data = df[df["City"] == city]

# =========================
# MAIN DASHBOARD
# =========================
if menu == "Main Dashboard":

    st.subheader("📈 Livability Trend")

    fig, ax = plt.subplots()
    ax.plot(city_data["Year"], city_data["Livability_Index"], marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Livability Index")
    ax.set_title(f"{city} Livability Trend")

    st.pyplot(fig)

    # 🔮 Forecast
    st.subheader("🔮 Future Livability Forecast (2025–2030)")

    latest = city_data.sort_values("Year").iloc[-1]

    future_years = [2025, 2026, 2027, 2028, 2029, 2030]
    future_rows = []

    for year in future_years:
        temp = latest.copy()
        temp["Year"] = year
        future_rows.append(temp)

    future_df = pd.DataFrame(future_rows)

    future_X = future_df[feature_cols]
    future_df["Predicted_Livability"] = model.predict(future_X)

    st.dataframe(future_df[["Year", "Predicted_Livability"]])

    # 🟢 Classification
    st.subheader("🟢 Livability Classification")

    latest_score = future_df["Predicted_Livability"].iloc[0]
    label = classify(latest_score)

    st.write(f"Predicted Livability Class for {city} in 2025: **{label}**")

# =========================
# 🏆 RANKING PAGE
# =========================
elif menu == "City Livability Ranking":

    st.subheader("🏆 City Livability Ranking")

    ranking = (
        df.groupby("City")["Predicted_Livability"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.dataframe(ranking)

# =========================
# 📊 FEATURE IMPORTANCE PAGE
# =========================
elif menu == "Feature Importance Analysis":

    show_feature_importance(model, feature_cols)

# =========================
# 🎛️ POLICY SIMULATION PAGE
# =========================
elif menu == "Policy Simulation (What-If)":

    latest = city_data.sort_values("Year").iloc[-1]
    run_simulation(latest, model, feature_cols,city)
    