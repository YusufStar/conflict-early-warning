#!/usr/bin/env python3
"""
Train conflict early warning model: LR, XGBoost, LGBM, or LSTM.
Uses enriched panel (VIEWS, alliance, MID) when --use_extra_data.
"""
import argparse
import json
import joblib
from pathlib import Path

import numpy as np

from src.data import load_and_build_panel, build_enriched_panel
from src.config import PATHS as DATA_PATHS
from src.features import build_features, time_based_split, get_feature_columns
from src.targets import add_targets
from src.evaluate import evaluate_binary, evaluate_regression
from src.models import get_lr_pipeline, get_xgb_model, get_lgbm_model
from src.models.lstm import build_sequences, train_lstm, predict_lstm, SEQ_LEN_DEFAULT


def parse_args():
    p = argparse.ArgumentParser(description="Train conflict escalation predictor")
    p.add_argument("--data_path", type=str, default=None, help="Political Violence Excel; default from config")
    p.add_argument("--no_extra_data", action="store_true", help="Disable VIEWS, alliance, MID (default: use extra data)")
    p.add_argument("--target", type=str, choices=["binary", "regression"], default="binary")
    p.add_argument("--model", type=str, choices=["lr", "xgb", "lgbm", "lstm"], default="xgb")
    p.add_argument("--test_months", type=int, default=12)
    p.add_argument("--val_months", type=int, default=12)
    p.add_argument("--high_risk_percentile", type=float, default=75.0)
    p.add_argument("--lag_months", type=str, default="1,2,3,6,12", help="Comma-separated lags")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--lstm_seq_len", type=int, default=SEQ_LEN_DEFAULT)
    p.add_argument("--lstm_epochs", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lag_months = [int(x) for x in args.lag_months.split(",")]
    data_path = args.data_path or str(DATA_PATHS["political_violence"])
    use_extra_data = not args.no_extra_data

    if use_extra_data:
        panel = build_enriched_panel(
            data_path,
            views_path=DATA_PATHS.get("views_fatalities"),
            alliance_path=DATA_PATHS.get("alliance_member_yearly"),
            dyadic_mid_path=DATA_PATHS.get("dyadic_mid"),
        )
    else:
        panel = load_and_build_panel(data_path)
    panel_with_target, threshold = add_targets(
        panel,
        target_type=args.target,
        high_risk_percentile=args.high_risk_percentile,
    )
    target_col = "target_binary" if args.target == "binary" else "target_events"
    if target_col not in panel_with_target.columns:
        raise RuntimeError(f"Missing target column {target_col}")

    if args.model == "lstm":
        X_seq, y_seq, period_seq = build_sequences(
            panel_with_target,
            seq_len=args.lstm_seq_len,
            target_type=args.target,
            target_col=target_col,
        )
        periods = sorted(panel_with_target["period_index"].unique())
        n = len(periods)
        test_months = min(args.test_months, max(1, n // 3))
        val_months = min(args.val_months, max(1, (n - test_months) // 2))
        test_cut = periods[-test_months]
        val_cut = periods[-(test_months + val_months)]
        train_mask = period_seq < val_cut
        val_mask = (period_seq >= val_cut) & (period_seq < test_cut)
        test_mask = period_seq >= test_cut
        X_train, y_train = X_seq[train_mask], y_seq[train_mask]
        X_val, y_val = X_seq[val_mask], y_seq[val_mask]
        X_test, y_test = X_seq[test_mask], y_seq[test_mask]
        import torch
        model, _ = train_lstm(X_train, y_train, X_val, y_val, target_type=args.target, epochs=args.lstm_epochs)
        y_val_pred = predict_lstm(model, X_val)
        y_test_pred = predict_lstm(model, X_test)
        if args.target == "binary":
            y_val_prob = y_val_pred
            y_test_prob = y_test_pred
            y_val_pred = (y_val_pred >= 0.5).astype(int)
            y_test_pred = (y_test_pred >= 0.5).astype(int)
        else:
            y_val_prob = y_test_prob = None
        artifacts = {
            "model": model,
            "model_type": "lstm",
            "country_encoder": None,
            "feature_columns": None,
            "target_type": args.target,
            "threshold": threshold,
            "lag_months": [args.lstm_seq_len],
            "seq_len": args.lstm_seq_len,
            "use_extra_data": use_extra_data,
            "data_path": data_path,
            "views_path": str(DATA_PATHS.get("views_fatalities", "")) if use_extra_data else None,
            "alliance_path": str(DATA_PATHS.get("alliance_member_yearly", "")) if use_extra_data else None,
            "dyadic_mid_path": str(DATA_PATHS.get("dyadic_mid", "")) if use_extra_data else None,
        }
        joblib.dump(artifacts, out_dir / "artifacts.joblib")
    else:
        df, country_encoder = build_features(
            panel_with_target,
            lag_months=lag_months,
            add_rolling=True,
            add_month_dum=True,
            fit_country_encoder=True,
        )
        if target_col not in df.columns:
            raise RuntimeError(f"Missing target column {target_col}")
        train_df, val_df, test_df = time_based_split(
            df, test_months=args.test_months, val_months=args.val_months
        )
        feat_cols = get_feature_columns(df)
        if not feat_cols:
            raise RuntimeError("No feature columns found")
        X_train = train_df[feat_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feat_cols].values
        y_val = val_df[target_col].values
        X_test = test_df[feat_cols].values
        y_test = test_df[target_col].values
        if args.model == "lr":
            if args.target != "binary":
                raise ValueError("LR only supports --target binary")
            pipeline = get_lr_pipeline(args.target)
            pipeline.fit(X_train, y_train)
            model = pipeline
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1] if args.target == "binary" else None
            y_test_pred = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)[:, 1] if args.target == "binary" else None
        elif args.model == "lgbm":
            model = get_lgbm_model(args.target)
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1] if args.target == "binary" else None
            y_test_pred = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)[:, 1] if args.target == "binary" else None
        else:
            model = get_xgb_model(args.target)
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1] if args.target == "binary" else None
            y_test_pred = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)[:, 1] if args.target == "binary" else None
        artifacts = {
            "model": model,
            "model_type": "sklearn",
            "country_encoder": country_encoder,
            "feature_columns": feat_cols,
            "target_type": args.target,
            "threshold": threshold,
            "lag_months": lag_months,
            "use_extra_data": use_extra_data,
            "data_path": data_path,
            "views_path": str(DATA_PATHS.get("views_fatalities", "")) if use_extra_data else None,
            "alliance_path": str(DATA_PATHS.get("alliance_member_yearly", "")) if use_extra_data else None,
            "dyadic_mid_path": str(DATA_PATHS.get("dyadic_mid", "")) if use_extra_data else None,
        }
        joblib.dump(artifacts, out_dir / "artifacts.joblib")

    if args.target == "binary":
        val_metrics = evaluate_binary(y_val, y_val_pred, y_val_prob)
        test_metrics = evaluate_binary(y_test, y_test_pred, y_test_prob)
    else:
        val_metrics = evaluate_regression(y_val, y_val_pred)
        test_metrics = evaluate_regression(y_test, y_test_pred)

    metrics = {"val": val_metrics, "test": test_metrics}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Val:", val_metrics)
    print("Test:", test_metrics)
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
