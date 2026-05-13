# Daily Revenue Forecasting — VinUniversity Datathon 2026

## Project Overview
This project was built for **VinUniversity Datathon 2026 — The Gridbreakers**, where the objective is to forecast daily net revenue for a simulated Vietnamese fashion e-commerce business.

The forecasting task supports key business decisions such as:

- Inventory allocation
- Promotion planning
- Logistics capacity planning
- Daily revenue monitoring

## Business Problem
Given historical daily sales data from **04/07/2012 to 31/12/2022**, the goal is to forecast daily `Revenue` for the period **01/01/2023 to 01/07/2024**.

The model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Dataset Summary

| Item | Value |
|---|---:|
| Full observed period in notebook | 04/07/2012 – 01/07/2024 |
| Number of daily observations | 4,381 |
| Forecast/evaluation period | 01/01/2023 – 01/07/2024 |
| Number of forecast days | 548 |
| Average daily revenue | 4.16M |
| Maximum daily revenue | 20.91M |
| Average daily COGS | 3.58M |
| Maximum daily COGS | 16.54M |

## Feature Engineering
The notebook includes several groups of forecasting features:

### Calendar Features
- Day
- Month
- Year
- Quarter
- Day of week
- Week of year
- Weekend indicator

### Special Day Features
- Vietnamese holidays
- Double-day campaigns, such as 1/1, 2/2, 3/3, etc.

### Cyclical Features
- Month sine/cosine
- Day-of-week sine/cosine

### Time-Series Features
- Revenue lag 1
- Revenue lag 7
- Revenue lag 30
- Rolling mean 7
- Rolling mean 30
- Rolling max 30

## Model Experiments
Four regression models were trained and tuned using GridSearchCV:

- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor
- Random Forest Regressor

## Model Performance

| Model | CV MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| CatBoost | 214,729 | 71,392 | 99,711 | 0.9960 |
| LightGBM | 220,344 | 77,001 | 111,755 | 0.9950 |
| XGBoost | 206,771 | 83,127 | 115,220 | 0.9947 |
| Random Forest | 335,619 | 127,502 | 194,295 | 0.9849 |

## Best Model
The best-performing model in the notebook experiment is **CatBoost**.

Best hyperparameters:

```python
{
    "iterations": 1000,
    "l2_leaf_reg": 1,
    "learning_rate": 0.05
}
```

## Key Insights
CatBoost delivered the strongest forecasting performance, achieving:

- MAE: **71.4K**
- RMSE: **99.7K**
- R²: **0.9960**

Compared with other models, CatBoost reduced MAE by approximately:

- **7.3%** versus LightGBM
- **14.1%** versus XGBoost
- **44.0%** versus Random Forest

CatBoost also reduced RMSE by approximately:

- **10.8%** versus LightGBM
- **13.5%** versus XGBoost
- **48.7%** versus Random Forest

This suggests that boosting-based models are more effective than Random Forest for capturing non-linear revenue patterns, calendar effects, and lag-based time-series signals in daily e-commerce data.

## Feature Importance Insight
Feature importance results show that `COGS` is the most dominant predictor across tree-based models. This indicates that cost of goods sold is highly correlated with net revenue and provides strong predictive information.

Other important features include:

- Rolling maximum revenue over 30 days
- Week of year
- Rolling mean revenue
- Month cyclical features
- Revenue lag features

## Validation Note
The current notebook should be interpreted as an **experimental model comparison**.

For a production-ready forecasting pipeline, additional improvements are recommended:

1. Use `TimeSeriesSplit` or walk-forward validation instead of standard 5-fold cross-validation.
2. Generate lag and rolling features using only historical information available before the prediction date.
3. Use shifted rolling features, for example `Revenue.shift(1).rolling(7).mean()`, to avoid using the current target value.
4. Use recursive forecasting when future revenue values are unavailable.
5. Confirm whether `COGS` is available in the official test set. If not, COGS should not be used as a prediction feature.

## Business Value
The forecasting pipeline can help an e-commerce business:

- Estimate future revenue by day
- Prepare stock allocation before high-demand periods
- Plan campaigns around holiday and double-day effects
- Support logistics capacity planning
- Monitor revenue volatility and abnormal demand patterns

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Random Forest
