import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

def train_gb_model():

    # =========================
    # Load dataset (FIXED PATH)
    # =========================
    df = pd.read_csv("../data/made3.csv").dropna()

    # =========================
    # CREATE SAME TARGET (COPY LOGIC — NO TOUCH TO RF FILE)
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
    # FEATURES & TARGET
    # =========================
    target = "Livability_Index"

    feature_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols.remove(target)

    X = df[feature_cols]
    y = df[target]

    # =========================
    # TIME SPLIT (SAME AS RF)
    # =========================
    train_df = df[df["Year"] <= 2018]
    test_df = df[df["Year"] > 2018]

    X_train = train_df[feature_cols]
    y_train = train_df[target]

    X_test = test_df[feature_cols]
    y_test = test_df[target]

    # =========================
    # MODEL
    # =========================
    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    gb.fit(X_train, y_train)

    # =========================
    # EVALUATION
    # =========================
    y_pred = gb.predict(X_test)

    print("R2 Score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    joblib.dump(gb, "../models/gb_model.pkl")
    joblib.dump(feature_cols, "../models/gb_feature_columns.pkl")

print("Gradient Boosting model saved!")

# CALL FUNCTION
if __name__ == "__main__":
    train_gb_model()