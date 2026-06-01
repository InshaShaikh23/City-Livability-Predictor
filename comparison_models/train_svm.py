import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

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
# FEATURE SET (SAME AS RF)
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
    "Literacy_Rate_%",
    "Internet_Penetration_%",
    "Crime_Rate_per_100k",
    "Average_AQI"
]

target = "Livability_Index"

# =========================
# TIME-BASED SPLIT (IMPORTANT)
# =========================
train_df = df[df["Year"] <= 2018]
test_df = df[df["Year"] > 2018]

X_train = train_df[feature_cols]
y_train = train_df[target]

X_test = test_df[feature_cols]
y_test = test_df[target]

# =========================
# SCALE DATA (MANDATORY FOR SVM)
# =========================
scaler_svm = StandardScaler()

X_train_scaled = scaler_svm.fit_transform(X_train)
X_test_scaled = scaler_svm.transform(X_test)

# =========================
# SVM MODEL
# =========================
model = SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=0.1
)

model.fit(X_train_scaled, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test_scaled)

print("🔥 SVM (Time-based Split) Results")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "../models/livability_svm.pkl")
joblib.dump(scaler_svm, "../models/svm_scaler.pkl")
joblib.dump(feature_cols, "../models/svm_features.pkl")

print("SVM model saved!")