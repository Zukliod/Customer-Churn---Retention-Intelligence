from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.churn_pipeline import explain_customer, generate_demo_data, train_models
from src.decision_engine import build_retention_plan

st.set_page_config(page_title="Churn Intelligence", page_icon="📉", layout="wide")


@st.cache_data
def load_demo() -> pd.DataFrame:
    return generate_demo_data()


@st.cache_resource
def train(data: pd.DataFrame):
    return train_models(data)


st.title("Customer Churn & Retention Intelligence")
st.caption("Predict who may leave, understand why, and prioritize economically sensible retention actions.")

with st.sidebar:
    st.header("Data and assumptions")
    upload = st.file_uploader("Upload Telco churn CSV", type=["csv"])
    success_rate = st.slider("Retention success rate", 0.05, 0.90, 0.35, 0.05)
    campaign_cost = st.number_input("Campaign cost per customer ($)", 0.0, 500.0, 25.0, 5.0)
    st.info("The demo dataset is synthetic. Replace it with the public Telco dataset for research results.")

data = pd.read_csv(upload) if upload is not None else load_demo()
try:
    result = train(data)
except ValueError as error:
    st.error(str(error))
    st.stop()

plan = build_retention_plan(result.scored_customers, success_rate, campaign_cost)
high_risk = int((plan["Risk"] == "High").sum())
contactable = int(plan["Contact"].sum())
revenue_at_risk = float(plan.loc[plan["Risk"].isin(["Medium", "High"]), "CustomerValue"].sum())

tab_overview, tab_customers, tab_retention, tab_model = st.tabs(
    ["Overview", "Customers", "Retention Center", "Model Performance"]
)

with tab_overview:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Customers", f"{len(plan):,}")
    metric_cols[1].metric("High-risk", f"{high_risk:,}")
    metric_cols[2].metric("Recommended contacts", f"{contactable:,}")
    metric_cols[3].metric("Value at risk", f"${revenue_at_risk:,.0f}")
    left, right = st.columns(2)
    with left:
        st.subheader("Risk distribution")
        st.plotly_chart(px.histogram(plan, x="Risk", color="Risk", category_orders={"Risk": ["Low", "Medium", "High"]}), use_container_width=True)
    with right:
        st.subheader("Churn by contract")
        contract_rates = plan.groupby("Contract", observed=False)["ChurnProbability"].mean().reset_index()
        st.plotly_chart(px.bar(contract_rates, x="Contract", y="ChurnProbability", labels={"ChurnProbability": "Average churn probability"}), use_container_width=True)
    st.subheader("Top model drivers")
    st.dataframe(result.feature_importance, hide_index=True, use_container_width=True)

with tab_customers:
    st.subheader("Customer risk list")
    risk_filter = st.multiselect("Risk filter", ["Low", "Medium", "High"], default=["Medium", "High"])
    view = plan[plan["Risk"].isin(risk_filter)] if risk_filter else plan.iloc[0:0]
    st.dataframe(
        view[["customerID", "Contract", "tenure", "MonthlyCharges", "ChurnProbability", "Risk", "CustomerValue"]]
        .assign(ChurnProbability=lambda frame: (frame["ChurnProbability"] * 100).round(1).astype(str) + "%"),
        hide_index=True, use_container_width=True,
    )
    if not view.empty:
        selected_id = st.selectbox("Inspect customer", view["customerID"].tolist())
        customer = plan.loc[plan["customerID"] == selected_id].iloc[0]
        st.subheader(f"Customer detail: {selected_id}")
        st.write(f"**{customer['Risk']} risk** · estimated churn probability **{customer['ChurnProbability']:.1%}**")
        for reason in explain_customer(customer):
            st.warning(reason)
        st.success(f"Suggested action: {customer['RecommendedAction']}")

with tab_retention:
    st.subheader("Cost-aware retention queue")
    st.caption("A contact is recommended only when expected benefit is positive and risk is Medium or High.")
    queue = plan[plan["Contact"]].copy()
    st.metric("Estimated net benefit of queue", f"${queue['NetBenefit'].sum():,.0f}")
    st.dataframe(
        queue[["customerID", "Risk", "ChurnProbability", "CustomerValue", "RecommendedAction", "NetBenefit"]]
        .assign(ChurnProbability=lambda frame: (frame["ChurnProbability"] * 100).round(1).astype(str) + "%"),
        hide_index=True, use_container_width=True,
    )

with tab_model:
    st.subheader("Model comparison")
    st.dataframe(result.metrics.style.format({column: "{:.3f}" for column in result.metrics.columns if column != "Model"}), hide_index=True, use_container_width=True)
    st.write("The selected model is the row ranked first by F1, then ROC-AUC. Class weights are used to avoid ignoring churners.")
    st.subheader("Test-set churn distribution")
    st.bar_chart(result.test_data[TARGET].value_counts().rename({0: "Stayed", 1: "Churned"}))
