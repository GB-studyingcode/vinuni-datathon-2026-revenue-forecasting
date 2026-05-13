# Model Card — Daily Revenue Forecasting

## Model Objective

Forecast daily net revenue for a Vietnamese fashion e-commerce business.

## Target Variable

- `Revenue`: daily net revenue.

## Main Features

- Calendar features: day, month, year, quarter, day of week, week of year.
- Business calendar features: weekend, holiday, double-day.
- Cyclical features: month sine/cosine, weekday sine/cosine.
- Time-series features: revenue lags and rolling statistics.
- Known future variable: `COGS`, if available in the submission structure.

## Validation Strategy

Time-based holdout validation is used to avoid leakage. The default validation period is 2022.

## Metrics

- MAE
- RMSE
- R²

## Risks and Limitations

- Recursive forecasting can accumulate errors over long horizons.
- If additional operational data is not integrated, model performance may miss promotion, inventory, or traffic-driven demand changes.
- Public leaderboard results may differ from local validation if the future period has structural changes.

## Next Development Steps

- Add rolling-origin backtesting.
- Compare multiple model families.
- Add SHAP or feature importance explanation.
- Link forecast insights to business actions.
