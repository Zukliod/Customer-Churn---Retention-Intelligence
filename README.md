# Customer Churn & Retention Decision Intelligence

An end-to-end Streamlit project based on the supplied customer churn presentation
and research synopsis. It predicts churn, explains risk, and recommends retention
actions using expected net benefit:

`Net Benefit = P(churn) × Customer Value × Success Rate − Campaign Cost`

## Features

- Built-in demo data shaped like the IBM Telco Customer Churn dataset
- CSV upload for a real Telco Customer Churn file
- Feature engineering for tenure stage, spend tier, service count, support/security flags, and customer value
- Logistic Regression baseline and balanced Random Forest comparison
- Stratified train/test split with cross-validation
- Risk buckets: Low, Medium, and High
- Plain-language per-customer explanations
- Cost-aware retention center with configurable success rate and campaign cost
- Interactive dashboard, customer table, customer detail view, and model metrics

## Run

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The app starts with generated demo data. Use the sidebar upload control to load
`WA_Fn-UseC_-Telco-Customer-Churn.csv` or another compatible CSV. The expected
target column is `Churn` with values such as `Yes`/`No`.

## Project layout

```text
app.py                    Streamlit user interface
src/churn_pipeline.py     Data preparation, training, scoring, explanations
src/decision_engine.py    Retention economics and action recommendations
tests/test_project.py     Core pipeline and decision-engine tests
```

## Important limitation

The default dataset is synthetic and the retention success rate and campaign
cost are assumptions. Model associations are not causal evidence; real
campaign outcomes should be used before deploying offers.

