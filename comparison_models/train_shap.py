import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../data/made3.csv").dropna()

# =========================
# DROP NOISY COLUMNS
# =========================
drop_cols = [
    "IPC_Crime_Cases",
    "MSME_Registered_Units",
    "Population_Density_per_sqkm"
]

df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# =========================
# CREATE LIVABILITY INDEX
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
# MANUAL FEATURE SELECTION (OPTIMIZED)
# =========================
feature_cols = [
    "Education_Index",
    "Infrastructure_Index",
    "Healthcare_Index",
    "Urbanization_Score",
    "Environment_Infrastructure_Interaction",
    "Education_Economic_Interaction",
    "Health_Safety_Interaction",

    "Electricity_Access_%",
    "PMAY_Average_Housing_Cost_Lakhs",
    "Population_Growth_%",
    "Hospitals_per_100k",
    "Affordability_Index",
    "Toilet_Access_%",
    "Unemployment_Rate_%",
    "Beds_per_1000",
    "Student_Teacher_Ratio",

    # 🔥 critical raw features
    "Literacy_Rate_%",
    "Internet_Penetration_%",
    "Crime_Rate_per_100k",
    "Average_AQI"
]

target = "Livability_Index"

# =========================
# PREPARE DATA
# =========================
X = df[feature_cols]
y = df[target]

# =========================
# TRAIN TEST SPLIT (BEST)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL (OPTIMIZED RF)
# =========================
model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

print("🔥 Manual Feature RF Results")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "../models/livability_manual_rf.pkl")
joblib.dump(feature_cols, "../models/manual_features.pkl")

print("Manual feature RF model saved!")