#!/usr/bin/env python3
"""
Predict next-period risk or event count using saved model and latest data.
"""
import argparse
import joblib
from pathlib import Path

import pandas as pd

from src.data import load_and_build_panel
from src.features import build_features, get_feature_columns


def get_predictions(
    data_path: str = "Political Violence Events by Country Mar 2026.xlsx",
    artifacts_path: str = "outputs/artifacts.joblib",
) -> tuple[list[dict], str]:
    """Return (list of prediction dicts, target_type)."""
    artifacts = joblib.load(artifacts_path)
    model = artifacts["model"]
    target_type = artifacts["target_type"]

    if artifacts.get("model_type") == "lstm":
        from src.models.lstm import get_latest_sequences, predict_lstm
        seq_len = artifacts.get("seq_len", 12)
        panel = load_and_build_panel(data_path)
        X, countries = get_latest_sequences(panel, seq_len)
        if not countries:
            raise RuntimeError("No country sequences for prediction")
        pred = predict_lstm(model, X)
        rows = []
        for c, p in zip(countries, pred):
            row = {"country": c}
            if target_type == "binary":
                row["risk_score"] = float(p)
                row["high_risk"] = 1 if p >= 0.5 else 0
            else:
                row["predicted_events"] = float(p)
            rows.append(row)
        return rows, target_type
    else:
        country_encoder = artifacts["country_encoder"]
        feature_columns = artifacts["feature_columns"]
        lag_months = artifacts.get("lag_months", [1, 2, 3, 6, 12])
        panel = load_and_build_panel(data_path)
        df, _ = build_features(
            panel,
            lag_months=lag_months,
            add_rolling=True,
            add_month_dum=True,
            country_encoder=country_encoder,
            fit_country_encoder=False,
        )
        latest_period = df["period_index"].max()
        pred_df = df[df["period_index"] == latest_period].copy()
        if pred_df.empty:
            raise RuntimeError("No rows at latest period for prediction")
        feat_cols = [c for c in feature_columns if c in pred_df.columns]
        X = pred_df[feat_cols].values
        if target_type == "binary":
            pred_df["risk_score"] = model.predict_proba(X)[:, 1]
            pred_df["high_risk"] = model.predict(X)
        else:
            pred_df["predicted_events"] = model.predict(X)
        cols = ["country", "year", "month", "period_index"]
        cols += ["risk_score", "high_risk"] if target_type == "binary" else ["predicted_events"]
        return pred_df[cols].to_dict("records"), target_type


def parse_args():
    p = argparse.ArgumentParser(description="Predict conflict risk / event count")
    p.add_argument("--data_path", type=str, default="Political Violence Events by Country Mar 2026.xlsx")
    p.add_argument("--artifacts_path", type=str, default="outputs/artifacts.joblib")
    p.add_argument("--out_path", type=str, default="outputs/predictions.csv")
    return p.parse_args()


def main():
    args = parse_args()
    rows, target_type = get_predictions(args.data_path, args.artifacts_path)
    pd.DataFrame(rows).to_csv(args.out_path, index=False)
    print("Predictions saved to", args.out_path)


if __name__ == "__main__":
    main()
