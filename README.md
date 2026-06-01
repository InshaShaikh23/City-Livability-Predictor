# Indian City Livability Score Prediction using Machine Learning

## How to Run the Project

### Prerequisites

Make sure Python 3.10+ is installed on your system.

### Step 1: Install Required Libraries

Open a terminal in the project root directory and run:

```
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Step 2: Install Streamlit

```bash
pip install streamlit
```

### Step 3: Navigate to the Source Directory

```
cd src
```

### Step 4: Train the Machine Learning Model

Run the training script to preprocess the data and train the Random Forest model:

```
python train.py
```

This will:

* Load and preprocess the dataset
* Train the machine learning model
* Save the trained model for future predictions

### Step 5: Launch the Dashboard

Start the Streamlit dashboard using:

```
python -m streamlit run app.py
```

or

```
streamlit run app.py
```

### Step 6: Open in Browser

After execution, Streamlit will automatically provide a local URL similar to:

```
http://localhost:8501
```

Open the URL in your browser to access the Livability Prediction Dashboard.

---

## Project Structure

```text
City-Livability-Classifier/
│
├── data/
│   └── made3.csv
│
├── models/
│   └── livability_model_v3.pkl
│   └──feature_columns_v3.pkl
│
├── src/
│   ├── train3.py
│   ├── app3.py
│   └── feature_importance.py
│   └──simulation.py
│
└── README.md
```

### Dashboard Features

* Predict Livability Score
* Classify Cities into Low, Medium, and High Livability Categories
* Visualize Feature Importance
* SHAP-based Explainability Analysis
* Interactive Urban Planning Insights

------------------------------------------------------------------------------------------------------------------------------------------

## Project Overview

The Indian City Livability Score Prediction System is a machine learning-based decision support framework designed to predict and evaluate the livability of Indian cities. The project analyzes various urban indicators such as education, healthcare, infrastructure, urbanization, and environmental conditions to generate a livability score and classify cities into qualitative categories.

The primary objective is not only to predict livability scores but also to identify the key factors influencing urban livability, enabling policymakers and urban planners to make informed decisions for sustainable city development.

---

## Problem Statement

Most cities do not have a directly available livability score. Therefore, this project creates a custom Livability Index using multiple urban indicators and employs machine learning models to predict city livability.

The system aims to answer:

* How livable is a city?
* Which factors most influence livability?
* How can policymakers improve city conditions?
* How can future livability trends be predicted using historical data?

---

## Dataset Features

### Positive Indicators

Higher values indicate better livability.

Examples:

* Education Index
* Healthcare Index
* Infrastructure Index
* Urbanization Score

### Negative Indicators

Lower values indicate better livability.

Examples:

* Pollution
* Crime Rate
* Unemployment Rate

To maintain consistency, negative indicators are transformed using:

```python
df_neg = 1 - df_neg
```

After transformation, higher values indicate better livability for all indicators.

---

## Livability Index Calculation

Since no direct livability label was available, an index-based modeling approach was adopted.

### Formula

```python
df["Livability_Index"] = (
    df_pos.mean(axis=1) * 0.6 +
    df_neg.mean(axis=1) * 0.4
) * 100
```

### Explanation

1. Positive Indicators Average

```python
df_pos.mean(axis=1)
```

Calculates the average of all positive indicators.

2. Negative Indicators Average

```python
df_neg.mean(axis=1)
```

Calculates the average of transformed negative indicators.

3. Weighting Strategy

* Positive Indicators: 60%
* Negative Indicators: 40%

4. Scaling

```python
* 100
```

Converts the score into a 0–100 scale for easier interpretation.

---

## Selected Features

Feature selection was performed using both Random Forest Feature Importance and SHAP analysis.

The most influential features identified were:

* Education_Index
* Infrastructure_Index
* Healthcare_Index
* Urbanization_Score
* Environment_Infrastructure_Interaction

---

## Machine Learning Models

### 1. Random Forest Regressor

Performance:

* R² Score: 0.7055
* MAE: 3.1930

Hyperparameters:

```python
RandomForestRegressor(
    n_estimators=700,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)
```

Parameter Description:

* n_estimators = Number of trees in the forest
* max_depth = Maximum depth of each tree
* random_state = Ensures reproducibility
* n_jobs = -1 uses all available CPU cores

---

### 2. Gradient Boosting Regressor

Performance:

* R² Score: 0.7705
* MAE: 2.7750

Advantages:

* Higher predictive accuracy
* Captures complex patterns

Limitations:

* More prone to overfitting
* Sensitive to hyperparameter tuning

---

### 3. Support Vector Machine (SVM)

Performance:

* R² Score: 0.8981
* MAE: 1.7329

Advantages:

* Highest prediction accuracy among tested models

Limitations:

* Black-box model
* Difficult to interpret
* Limited explainability for policy analysis

---

### 4. SHAP-Optimized Random Forest

Performance:

* R² Score: 0.6796
* MAE: 3.4559

Purpose:

Used to improve model explainability and understand feature contributions.

---

## Explainable AI using SHAP

### Random Forest Feature Importance

Provides ranking of features using:

```python
model.feature_importances_
```

However, it only indicates relative importance.

### SHAP (SHapley Additive Explanations)

SHAP provides:

* Feature ranking
* Feature contribution magnitude
* Direction of influence

It explains:

* Which feature affected the prediction
* Whether it increased or decreased the livability score
* By how much it influenced the result

This makes SHAP highly suitable for policy-oriented decision support systems.

---

## Time-Based Train-Test Split

### Strategy

Training Data:

```text
Year ≤ 2018
```

Testing Data:

```text
Year > 2018
```

### Why Time-Based Splitting?

#### 1. Real-World Prediction

The system learns from historical data and predicts future livability trends.

#### 2. Preserves Temporal Patterns

The dataset contains:

* Year information
* Urban growth patterns
* Economic changes
* Environmental changes

A time-based split preserves these trends.

#### 3. Prevents Data Leakage

Random splitting may allow future data to appear in training data.

Example:

Training: 2020

Testing: 2015

This introduces unrealistic knowledge of future events.

Time-based splitting avoids this issue and provides more reliable evaluation.

---

## Livability Classification

Predicted livability scores are categorized into:

| Category | Score Range |
| -------- | ----------- |
| Low      | < 45        |
| Medium   | 45 – 55     |
| High     | ≥ 55        |

### Reason for Threshold Selection

Analysis of the dataset showed that most city livability scores cluster around 50.

Therefore:

* Below 45 indicates poor livability
* 45–55 indicates average livability
* Above 55 indicates high livability

---

## Why Random Forest Was Selected

Although SVM and Gradient Boosting achieved higher accuracy, Random Forest was chosen as the primary model.

### 1. Interpretability

Random Forest:

* Provides feature importance
* Works effectively with SHAP
* Easy to explain results

SVM:

* Black-box model

Gradient Boosting:

* More difficult to interpret

---

### 2. Stability and Robustness

Random Forest:

* Trees built independently
* Lower overfitting risk
* Robust against noisy data

Gradient Boosting:

* Sequential learning
* Higher risk of overfitting

SVM:

* Sensitive to kernel selection and parameter tuning

---

### 3. Project Objectives

This project focuses on:

* Livability prediction
* Feature importance analysis
* Policy simulation
* Decision support

These objectives require explainable predictions rather than only maximizing accuracy.

---

### 4. Literature Support

Most recent urban livability and smart city research studies utilize:

* Random Forest
* Tree-based Ensemble Models

Random Forest offers an effective balance between:

* Accuracy
* Explainability
* Reliability

making it suitable for urban planning applications.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* SHAP
* Matplotlib
* Seaborn
* Streamlit

---

## Future Enhancements

* Integration of real-time government datasets
* GIS-based spatial visualization
* Smart city ranking dashboard
* Policy impact simulation
* Deep Learning and XGBoost comparison
* Interactive decision-support tools for urban planners

---

## Conclusion

The proposed system demonstrates how machine learning and explainable AI can be used to assess urban livability. While SVM achieved the highest predictive accuracy, Random Forest was selected due to its superior interpretability, stability, and suitability for decision-support systems. The framework provides valuable insights for policymakers, researchers, and urban planners working toward sustainable and livable cities.
