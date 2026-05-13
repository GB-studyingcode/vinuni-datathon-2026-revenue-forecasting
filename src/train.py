from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import lightgbm as lgb
import joblib

from features import build_training_frame
from modeling import evaluate, get_feature_columns, time_holdout_split


def main(train_path: str, model_path: str, holdout_start: str):
    sales = pd.read_csv(train_path, parse_dates=["Date"])
    frame = build_training_frame(sales, target_col="Revenue")

    train_df, valid_df = time_holdout_split(frame, holdout_start=holdout_start)
    features = get_feature_columns(frame, target_col="Revenue")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(
        train_df[features],
        train_df["Revenue"],
        eval_set=[(valid_df[features], valid_df["Revenue"])],
        eval_metric="mae",
    )

    preds = model.predict(valid_df[features])
    metrics = evaluate(valid_df["Revenue"], preds)
    print(metrics)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "metrics": metrics}, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/raw/sales.csv")
    parser.add_argument("--model-path", default="models/lightgbm_revenue.pkl")
    parser.add_argument("--holdout-start", default="2022-01-01")
    args = parser.parse_args()
    main(args.train_path, args.model_path, args.holdout_start)
