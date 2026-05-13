from __future__ import annotations

import numpy as np
import pandas as pd
import holidays


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Create deterministic date features available for both train and test."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    out["day"] = out[date_col].dt.day
    out["month"] = out[date_col].dt.month
    out["year"] = out[date_col].dt.year
    out["quarter"] = out[date_col].dt.quarter
    out["day_of_week"] = out[date_col].dt.dayofweek
    out["week_of_year"] = out[date_col].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["is_month_start"] = out[date_col].dt.is_month_start.astype("int8")
    out["is_month_end"] = out[date_col].dt.is_month_end.astype("int8")
    out["is_double_day"] = (out["day"] == out["month"]).astype("int8")

    years = range(out[date_col].dt.year.min(), out[date_col].dt.year.max() + 1)
    vn_holidays = holidays.VN(years=years)
    out["is_holiday"] = out[date_col].isin(vn_holidays).astype("int8")

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Revenue",
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Create lag and rolling features using only past target values."""
    lags = lags or [1, 7, 14, 30]
    windows = windows or [7, 14, 30]
    out = df.sort_values("Date").copy()

    for lag in lags:
        out[f"revenue_lag_{lag}"] = out[target_col].shift(lag)

    for window in windows:
        shifted = out[target_col].shift(1)
        out[f"revenue_roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"revenue_roll_std_{window}"] = shifted.rolling(window).std()
        out[f"revenue_roll_max_{window}"] = shifted.rolling(window).max()
        out[f"revenue_roll_min_{window}"] = shifted.rolling(window).min()

    return out


def build_training_frame(df: pd.DataFrame, target_col: str = "Revenue") -> pd.DataFrame:
    """Full feature pipeline for historical training data."""
    out = add_calendar_features(df)
    out = add_lag_rolling_features(out, target_col=target_col)
    return out.dropna().reset_index(drop=True)
