from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float


def evaluate(y_true, y_pred) -> Metrics:
    """Evaluate forecasts with the competition metrics."""
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
    )


def time_holdout_split(
    df: pd.DataFrame,
    holdout_start: str = "2022-01-01",
    date_col: str = "Date",
):
    """Split historical data into train/validation by date to avoid time leakage."""
    train_df = df[df[date_col] < holdout_start].copy()
    valid_df = df[df[date_col] >= holdout_start].copy()
    return train_df, valid_df


def get_feature_columns(df: pd.DataFrame, target_col: str = "Revenue") -> list[str]:
    excluded = {"Date", target_col, "is_test"}
    return [c for c in df.columns if c not in excluded]
