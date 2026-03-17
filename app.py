#!/usr/bin/env python3
"""Single-page web UI for conflict early warning predictions."""
import os
from pathlib import Path

from flask import Flask, jsonify, render_template

from predict import get_predictions

app = Flask(__name__, template_folder="templates")
ROOT = Path(__file__).resolve().parent
DATA_PATH = os.environ.get("DATA_PATH", str(ROOT / "Political Violence Events by Country Mar 2026.xlsx"))
ARTIFACTS_PATH = os.environ.get("ARTIFACTS_PATH", str(ROOT / "outputs" / "artifacts.joblib"))


def _float(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return x


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predictions")
def api_predictions():
    try:
        rows, target_type = get_predictions(DATA_PATH, ARTIFACTS_PATH)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    # Normalize for JSON (numpy/pandas types)
    out = []
    for r in rows:
        row = {"country": str(r["country"])}
        if target_type == "binary":
            row["risk_score"] = _float(r.get("risk_score", 0))
            row["high_risk"] = int(r.get("high_risk", 0))
        else:
            row["predicted_events"] = _float(r.get("predicted_events", 0))
        out.append(row)
    return jsonify({"target_type": target_type, "predictions": out})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
