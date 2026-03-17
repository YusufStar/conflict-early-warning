# Conflict Early Warning / Escalation Predictor

**"AI that predicts where conflict will escalate next."**

Country-level **next month risk score** or **conflict escalation probability**, trained on historical events plus VIEWS fatalities, COW alliances, and MIDs (default).

## Data (default: all used)

- **Base:** `data/Political Violence Events by Country Mar 2026.xlsx` (ACLED: COUNTRY, MONTH, YEAR, EVENTS).
- **Extra (default on):** `data/VIEWS Fatalities Data 2026.csv`, `data/version4.1_csv/alliance_v4.1_by_member_yearly.csv`, `data/dyadic_mid_4.03.csv`. Paths in `src/config.py`.
- To train without extra data: `python train.py --no_extra_data --model lr --target binary`.
- See `docs/DATA_PLAN.md` for integration details.

## Models

- **Target:** Next-period "high risk" (binary) or event count (regression).
- **Features:** Event lags, trend, seasonality, country dummies, VIEWS fatalities, alliance count, MID count.
- **Classical:** Logistic Regression (`lr`), XGBoost (`xgb`), LightGBM (`lgbm`, Tweedie for regression).
- **Time-series:** LSTM (`lstm`).

## Setup (macOS / Linux)

```bash
cd ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost and LightGBM may require: `brew install libomp`.

## Usage

**Train** (uses VIEWS + alliance + MID by default)

```bash
python train.py --model lr --target binary
python train.py --model xgb --target binary
python train.py --model lgbm --target binary
python train.py --model lstm --target binary --lstm_epochs 50

# Without extra data
python train.py --no_extra_data --model lr --target binary
```

Options: `--data_path`, `--no_extra_data`, `--test_months`, `--val_months`, `--high_risk_percentile`, `--lag_months`, `--out_dir`, `--lstm_seq_len`, `--lstm_epochs`.

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

## Tests

```bash
pytest tests/ -v
```

## License

MIT. See [LICENSE](LICENSE).
