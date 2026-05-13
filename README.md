# VinUniversity Datathon 2026 — Revenue Forecasting for Fashion E-commerce

## 1. Project Overview

This repository contains my solution framework for **Datathon 2026 — The Gridbreakers**, a data science competition hosted by VinUniversity. The business scenario simulates a Vietnamese fashion e-commerce company that needs accurate **daily net revenue forecasting** to support inventory allocation, promotion planning, and logistics operations.

The main task is to forecast daily `Revenue` for the period **01/01/2023–01/07/2024** using historical business data from **04/07/2012–31/12/2022**.

## 2. Business Problem

Accurate daily revenue forecasting helps the business answer three practical questions:

- **Inventory:** how much stock should be prepared before demand peaks?
- **Promotion:** when should the company launch campaigns or double-day offers?
- **Logistics:** how should warehouse and delivery capacity be planned across the network?

This project is designed not only as a Kaggle submission, but also as a portfolio case study showing the full workflow from data understanding to model validation and business interpretation.

## 3. Data Sources

The original competition dataset contains 15 CSV files grouped into four layers:

| Data Layer | Examples | Role in Forecasting |
|---|---|---|
| Master | products, customers, promotions, geography | enrich customer/product/market context |
| Transaction | orders, order details, payments, shipping, returns, reviews | capture sales behavior and operational signals |
| Analytical | daily sales/revenue | main target series |
| Operational | monthly inventory, daily website traffic | explain demand and supply-side constraints |

> Note: raw competition data is not committed to this repository. Place downloaded CSV files in `data/raw/`.

## 4. Evaluation Metrics

The competition evaluates submissions using three metrics:

| Metric | Meaning | Optimization Direction |
|---|---|---|
| MAE | average absolute forecast error | lower is better |
| RMSE | penalizes large forecast errors more strongly | lower is better |
| R² | proportion of target variance explained by the model | higher is better |

## 5. Modeling Approach

Current baseline workflow:

1. Load historical daily sales data and sample submission structure.
2. Conduct exploratory data analysis on `Revenue` and `COGS` trends.
3. Build time-based features: day, month, quarter, weekday, weekend, holiday, double-day, cyclic features.
4. Build lag and rolling features from historical revenue.
5. Use time-based validation to avoid data leakage.
6. Train tree-based regression models such as LightGBM, XGBoost, CatBoost, and Random Forest.
7. Generate Kaggle-compatible submission file.

## 6. Repository Structure

```text
vinuni-datathon-gridbreakers/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/              # Put Kaggle CSV files here; not committed
│   └── processed/        # Processed feature tables; not committed
├── notebooks/
│   └── Forecasting_VinUni_original.ipynb
├── src/
│   ├── features.py       # Calendar, lag, rolling feature engineering
│   ├── modeling.py       # Validation split and metric functions
│   ├── train.py          # Model training script
│   └── predict.py        # Recursive future forecasting script
├── models/               # Trained models; not committed
├── outputs/              # Kaggle submission files; not committed
├── reports/
│   └── model_card.md
└── assets/               # Charts/screenshots for README
```

## 7. How to Run

### Step 1 — Clone repository

```bash
git clone https://github.com/<your-username>/vinuni-datathon-gridbreakers.git
cd vinuni-datathon-gridbreakers
```

### Step 2 — Create environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### Step 3 — Add data

Place these files in `data/raw/`:

```text
sales.csv
sample_submission.csv
```

### Step 4 — Train model

```bash
python src/train.py --train-path data/raw/sales.csv --model-path models/lightgbm_revenue.pkl
```

### Step 5 — Create submission

```bash
python src/predict.py \
  --train-path data/raw/sales.csv \
  --sample-path data/raw/sample_submission.csv \
  --model-path models/lightgbm_revenue.pkl \
  --output-path outputs/submission.csv
```

## 8. Key Technical Notes

A key issue in time-series forecasting is **data leakage**. Future revenue is unknown during the Kaggle test period, so lag and rolling features for test dates should be generated recursively using previous predictions, not future actual revenue values.

This repository separates:

- deterministic future features: calendar, holiday, weekday, month, double-day;
- historical target features: lag and rolling revenue features;
- validation logic: time-based holdout rather than random split.

## 9. Planned Improvements

- Add EDA charts for trend, seasonality, holiday effects, and COGS–Revenue relationship.
- Add baseline comparisons: naive forecast, moving average, LightGBM, XGBoost, CatBoost.
- Add feature importance analysis and business interpretation.
- Add backtesting with multiple rolling validation folds.
- Integrate additional files such as promotions, website traffic, inventory, returns, and reviews.
- Add final `submission.csv` generation notebook for Kaggle.

## 10. Portfolio Positioning

This project demonstrates:

- time-series forecasting for business planning;
- feature engineering with calendar, lag, rolling, and holiday signals;
- leakage-aware validation design;
- practical use of machine learning models for demand and revenue forecasting;
- ability to translate model output into inventory, promotion, and logistics decisions.
