from __future__ import annotations

import pandas as pd


def recommend_action(customer: pd.Series) -> str:
    if customer.get("Contract") == "Month-to-month":
        return "Offer a contract upgrade incentive"
    if customer.get("OnlineSecurity") == "No" or customer.get("TechSupport") == "No":
        return "Offer security or support bundle"
    if customer.get("MonthlyCharges", 0) >= 80:
        return "Offer a targeted bill review"
    return "Send a loyalty check-in"


def build_retention_plan(
    scored: pd.DataFrame,
    success_rate: float = 0.35,
    campaign_cost: float = 25.0,
) -> pd.DataFrame:
    plan = scored.copy()
    plan["ExpectedBenefit"] = (
        plan["ChurnProbability"] * plan["CustomerValue"] * success_rate
    )
    plan["CampaignCost"] = campaign_cost
    plan["NetBenefit"] = plan["ExpectedBenefit"] - campaign_cost
    plan["RecommendedAction"] = plan.apply(recommend_action, axis=1)
    plan["Contact"] = (plan["NetBenefit"] > 0) & (plan["Risk"].isin(["Medium", "High"]))
    return plan.sort_values(["Contact", "NetBenefit"], ascending=[False, False])

