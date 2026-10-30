from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMNS = {"customerID", "CustomerID"}
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [
    "TenureStage", "SpendTier", "ServiceCount", "SecurityOrSupport",
    "MonthToMonth", "CustomerValue",
]


@dataclass
class TrainingResult:
    models: dict[str, Pipeline]
    metrics: pd.DataFrame
    scored_customers: pd.DataFrame
    feature_importance: pd.DataFrame
    test_data: pd.DataFrame


def generate_demo_data(n_rows: int = 7043, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contracts = rng.choice(["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.21, 0.24])
    tenure = rng.integers(0, 73, n_rows)
    monthly = np.round(rng.normal(70, 25, n_rows).clip(18, 140), 2)
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n_rows, p=[0.34, 0.23, 0.22, 0.21],
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_rows, p=[0.35, 0.45, 0.20])
    data = pd.DataFrame({
        "customerID": [f"DEMO-{i:05d}" for i in range(n_rows)],
        "gender": rng.choice(["Female", "Male"], n_rows),
        "SeniorCitizen": rng.choice([0, 1], n_rows, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n_rows),
        "Dependents": rng.choice(["Yes", "No"], n_rows, p=[0.30, 0.70]),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], n_rows, p=[0.90, 0.10]),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n_rows, p=[0.42, 0.48, 0.10]),
        "InternetService": internet,
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n_rows, p=[0.29, 0.51, 0.20]),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n_rows, p=[0.32, 0.48, 0.20]),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n_rows, p=[0.31, 0.49, 0.20]),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n_rows, p=[0.29, 0.51, 0.20]),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n_rows, p=[0.38, 0.42, 0.20]),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n_rows, p=[0.38, 0.42, 0.20]),
        "Contract": contracts,
        "PaperlessBilling": rng.choice(["Yes", "No"], n_rows, p=[0.60, 0.40]),
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
    })
    data["TotalCharges"] = np.round((monthly * tenure + rng.normal(0, 80, n_rows)).clip(0), 2)
    logit = (
        -2.05
        + 1.15 * (contracts == "Month-to-month")
        + 0.85 * (tenure < 12)
        + 0.55 * (payment == "Electronic check")
        + 0.35 * (monthly > 80)
        + 0.30 * (data["OnlineSecurity"] == "No")
        + 0.25 * data["SeniorCitizen"]
        - 0.55 * (contracts == "Two year")
        + rng.normal(0, 0.45, n_rows)
    )
    probability = 1 / (1 + np.exp(-logit))
    data[TARGET] = np.where(rng.random(n_rows) < probability, "Yes", "No")
    return data


def _clean_input(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].astype(str).str.strip()
    for column in NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["TotalCharges"] = frame["TotalCharges"].fillna(frame["MonthlyCharges"] * 0)
    frame["TotalCharges"] = frame["TotalCharges"].fillna(0)
    frame["tenure"] = frame["tenure"].fillna(0).clip(lower=0)
    frame["MonthlyCharges"] = frame["MonthlyCharges"].fillna(frame["MonthlyCharges"].median())
    frame["SeniorCitizen"] = frame["SeniorCitizen"].fillna(0)
    return frame


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = _clean_input(data)
    for column in CATEGORICAL_COLUMNS:
        if column not in frame:
            frame[column] = "Unknown"
    service_columns = [
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    frame["TenureStage"] = pd.cut(
        frame["tenure"], [-1, 12, 24, 48, 60, np.inf],
        labels=["0-12", "13-24", "25-48", "49-60", "60+"],
    ).astype(str)
    frame["SpendTier"] = pd.qcut(
        frame["MonthlyCharges"].rank(method="first"), 4,
        labels=["Low", "Medium", "High", "Very high"],
    ).astype(str)
    frame["ServiceCount"] = frame[service_columns].apply(
        lambda row: sum(value == "Yes" for value in row), axis=1
    )
    frame["SecurityOrSupport"] = (
        (frame["OnlineSecurity"] == "Yes") | (frame["TechSupport"] == "Yes")
    ).astype(int)
    frame["MonthToMonth"] = (frame["Contract"] == "Month-to-month").astype(int)
    frame["CustomerValue"] = (frame["MonthlyCharges"] * frame["tenure"]).round(2)
    return frame


def _preprocessor() -> ColumnTransformer:
    numeric = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
               "ServiceCount", "SecurityOrSupport", "MonthToMonth", "CustomerValue"]
    categorical = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                   "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                   "PaperlessBilling", "PaymentMethod", "TenureStage", "SpendTier"]
    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


def train_models(data: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> TrainingResult:
    frame = engineer_features(data)
    if TARGET not in frame:
        raise ValueError(f"Input data must contain a '{TARGET}' column.")
    y = frame[TARGET].map({"Yes": 1, "No": 0}).fillna(pd.to_numeric(frame[TARGET], errors="coerce"))
    if y.isna().any():
        raise ValueError("Churn must contain Yes/No or numeric 0/1 values.")
    x = frame[FEATURE_COLUMNS]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y.astype(int), test_size=test_size, stratify=y, random_state=seed
    )
    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", _preprocessor()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", _preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=250, max_depth=12, min_samples_leaf=3,
                class_weight="balanced", random_state=seed, n_jobs=-1,
            )),
        ]),
    }
    metric_rows: list[dict[str, Any]] = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metric_rows.append({
            "Model": name,
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1": f1_score(y_test, predictions, zero_division=0),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Accuracy": accuracy_score(y_test, predictions),
        })
    metrics = pd.DataFrame(metric_rows).sort_values(["F1", "ROC-AUC"], ascending=False)
    winner = str(metrics.iloc[0]["Model"])
    chosen = models[winner]
    all_probabilities = chosen.predict_proba(x)[:, 1]
    scored = frame.copy()
    scored["ChurnProbability"] = all_probabilities
    scored["Risk"] = pd.cut(
        scored["ChurnProbability"], [-0.01, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
    )
    importance = _importance_table(chosen, FEATURE_COLUMNS)
    test_data = x_test.copy()
    test_data[TARGET] = y_test.to_numpy()
    return TrainingResult(models, metrics, scored, importance, test_data)


def _importance_table(model: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        values = np.abs(estimator.coef_[0])
    table = pd.DataFrame({"Feature": names, "Importance": values})
    table["Feature"] = table["Feature"].str.replace(r"^(numeric|categorical)__", "", regex=True)
    return table.groupby("Feature", as_index=False)["Importance"].sum().sort_values(
        "Importance", ascending=False
    ).head(12)


def explain_customer(customer: pd.Series) -> list[str]:
    reasons: list[str] = []
    if customer.get("Contract") == "Month-to-month":
        reasons.append("Month-to-month contract means there is little commitment.")
    if customer.get("tenure", 0) < 12:
        reasons.append("The customer is in the higher-risk first year.")
    if customer.get("PaymentMethod") == "Electronic check":
        reasons.append("Electronic-check customers show elevated churn in this dataset.")
    if customer.get("OnlineSecurity") == "No" and customer.get("TechSupport") == "No":
        reasons.append("The customer has neither online security nor technical support.")
    if customer.get("MonthlyCharges", 0) >= 80:
        reasons.append("The monthly charge is relatively high.")
    if not reasons:
        reasons.append("No dominant risk driver was detected; monitor the account.")
    return reasons[:4]

