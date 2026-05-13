from __future__ import annotations

import argparse
import joblib
import pandas as pd

from features import add_calendar_features, add_lag_rolling_features


def recursive_forecast(history: pd.DataFrame, future: pd.DataFrame, model, feature_cols: list[str]) -> pd.DataFrame:
    """
    Forecast future Revenue day-by-day so lag/rolling features use previous predictions,
    not unknown future actuals.
    """
    history = history.copy().sort_values("Date")
    future = future.copy().sort_values("Date")
    predictions = []

    for _, row in future.iterrows():
        step = pd.DataFrame([row])
        step["Revenue"] = pd.NA
        combined = pd.concat([history, step], ignore_index=True)
        combined = add_calendar_features(combined)
        combined = add_lag_rolling_features(combined, target_col="Revenue")
        X_step = combined.tail(1)[feature_cols]
        y_hat = float(model.predict(X_step)[0])

        step["Revenue"] = y_hat
        history = pd.concat([history, step], ignore_index=True)
        predictions.append({"Date": row["Date"], "Revenue": y_hat})

    return pd.DataFrame(predictions)


def main(train_path: str, sample_path: str, model_path: str, output_path: str):
    artifact = joblib.load(model_path)
    model = artifact["model"]
    features = artifact["features"]

    history = pd.read_csv(train_path, parse_dates=["Date"])
    sample = pd.read_csv(sample_path, parse_dates=["Date"])
    forecast = recursive_forecast(history, sample, model, features)

    # Preserve any required sample_submission columns when possible.
    submission = sample.copy()
    submission["Revenue"] = forecast["Revenue"].values
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/raw/sales.csv")
    parser.add_argument("--sample-path", default="data/raw/sample_submission.csv")
    parser.add_argument("--model-path", default="models/lightgbm_revenue.pkl")
    parser.add_argument("--output-path", default="outputs/submission.csv")
    args = parser.parse_args()
    main(args.train_path, args.sample_path, args.model_path, args.output_path)
