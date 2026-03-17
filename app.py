#!/usr/bin/env python3
"""Single-page web UI for conflict early warning predictions."""
import os
import joblib
from pathlib import Path

from flask import Flask, jsonify, render_template

from predict import get_predictions
from src.config import PATHS as DATA_PATHS
from src.data import load_mid_history, load_alliance_count_by_year

app = Flask(__name__, template_folder="templates")
ROOT = Path(__file__).resolve().parent
DATA_PATH = os.environ.get("DATA_PATH", str(DATA_PATHS.get("political_violence", ROOT / "data" / "Political Violence Events by Country Mar 2026.xlsx")))
ARTIFACTS_PATH = os.environ.get("ARTIFACTS_PATH", str(ROOT / "outputs" / "artifacts.joblib"))

# Cached for country detail API
_mid_history = None
_alliance_by_country = None


def _float(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return x


def _get_mid_history():
    global _mid_history
    if _mid_history is None:
        p_mid = DATA_PATHS.get("dyadic_mid")
        p_all = DATA_PATHS.get("alliance_member_yearly")
        if p_mid and p_mid.exists() and p_all and p_all.exists():
            _mid_history = load_mid_history(p_mid, p_all)
        else:
            _mid_history = {}
    return _mid_history


def _get_alliance_latest():
    global _alliance_by_country
    if _alliance_by_country is None:
        p = DATA_PATHS.get("alliance_member_yearly")
        if p and p.exists():
            df = load_alliance_count_by_year(p)
            # latest year per country
            df = df.sort_values("year").groupby("country").last().reset_index()
            _alliance_by_country = dict(zip(df["country"], df["alliance_count"].astype(int)))
        else:
            _alliance_by_country = {}
    return _alliance_by_country


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predictions")
def api_predictions():
    try:
        rows, target_type = get_predictions(DATA_PATH, ARTIFACTS_PATH)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    out = []
    for r in rows:
        row = {"country": str(r["country"])}
        if target_type == "binary":
            row["risk_score"] = _float(r.get("risk_score", 0))
            row["high_risk"] = int(r.get("high_risk", 0))
        else:
            row["predicted_events"] = _float(r.get("predicted_events", 0))
        out.append(row)
    payload = {"target_type": target_type, "predictions": out}
    try:
        art = joblib.load(ARTIFACTS_PATH)
        if art.get("target_type") == "binary" and "threshold" in art:
            payload["high_risk_threshold"] = _float(art["threshold"])
    except Exception:
        pass
    return jsonify(payload)


@app.route("/api/country/<path:name>")
def api_country(name):
    """Country detail: risk + MID history (who fought whom, when) + alliance count."""
    name = name.strip()
    try:
        rows, target_type = get_predictions(DATA_PATH, ARTIFACTS_PATH)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    pred = next((r for r in rows if str(r.get("country", "")).strip() == name), None)
    if not pred:
        return jsonify({"error": "Country not found"}), 404
    out = {"country": name}
    if target_type == "binary":
        out["risk_score"] = _float(pred.get("risk_score", 0))
        out["high_risk"] = int(pred.get("high_risk", 0))
    else:
        out["predicted_events"] = _float(pred.get("predicted_events", 0))
    out["mid_history"] = _get_mid_history().get(name, [])
    out["alliance_count"] = _get_alliance_latest().get(name, 0)
    return jsonify(out)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
