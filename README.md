# Conflict Early Warning / Escalation Predictor

**"AI that predicts where conflict will escalate next."**

Country- or Admin1-level **next month/quarter risk score** or **conflict escalation probability**, trained on historical event/fatality counts.

## Data

- **Source:** `Political Violence Events by Country Mar 2026.xlsx`
- **Schema:** `COUNTRY`, `MONTH`, `YEAR`, `EVENTS` (country–month–year event counts).
- Optional: add regional weekly or Admin1 aggregates if available.

## Models

- **Target:** Next-period "high risk" (binary) or event count (regression).
- **Features:** Event lags, trend, seasonality, country dummies.
- **Classical:** Logistic Regression (`lr`), XGBoost (`xgb`).
- **Time-series:** LSTM (`lstm`) — country-level event sequences.

## Setup (macOS / Linux)

```bash
cd ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost may require: `brew install libomp`.

## Usage

**Train**

```bash
# Binary risk (default), XGBoost
python train.py --model xgb --target binary

# Logistic Regression
python train.py --model lr --target binary

# Regression (next-month event count)
python train.py --model xgb --target regression

# LSTM (time-series)
python train.py --model lstm --target binary --lstm_epochs 50
```

Options: `--data_path`, `--test_months`, `--val_months`, `--high_risk_percentile`, `--lag_months`, `--out_dir`, `--lstm_seq_len`, `--lstm_epochs`.

**Predict**

```bash
python predict.py --artifacts_path outputs/artifacts.joblib --out_path outputs/predictions.csv
```

## Outputs

- `outputs/artifacts.joblib` — trained model, encoder, feature list (or LSTM state).
- `outputs/metrics.json` — validation and test metrics (AUC, F1, MAE, etc.).
- `outputs/predictions.csv` — country-level risk score or predicted events.

## Web UI

Single-page site that shows model predictions (risk score / predicted events by country).

```bash
python app.py
```

Open http://127.0.0.1:5050 (or set `PORT`). Backend uses saved `outputs/artifacts.joblib` and the Excel data; UI supports search, “high risk only” filter, and sort.

## License

MIT. See [LICENSE](LICENSE).
