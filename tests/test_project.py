import pandas as pd

from src.churn_pipeline import generate_demo_data, train_models
from src.decision_engine import build_retention_plan


def test_training_produces_scores_and_metrics():
    result = train_models(generate_demo_data(500, seed=7))
    assert set(["Logistic Regression", "Random Forest"]) == set(result.metrics["Model"])
    assert len(result.scored_customers) == 500
    assert result.scored_customers["ChurnProbability"].between(0, 1).all()


def test_retention_net_benefit_uses_expected_value_formula():
    scored = pd.DataFrame([{
        "customerID": "A", "ChurnProbability": 0.8, "CustomerValue": 1000,
        "Risk": "High", "Contract": "Month-to-month", "OnlineSecurity": "No",
        "TechSupport": "No", "MonthlyCharges": 80,
    }])
    plan = build_retention_plan(scored, success_rate=0.5, campaign_cost=25)
    assert plan.loc[0, "ExpectedBenefit"] == 400
    assert plan.loc[0, "NetBenefit"] == 375
    assert bool(plan.loc[0, "Contact"])
