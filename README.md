# Conflict Early Warning / Escalation Predictor

**"AI that predicts where conflict will escalate next."**

Country-level **next month risk score** or **conflict escalation probability**, trained on historical events plus optional VIEWS fatalities, COW alliances, and MIDs.

## Data

- **Base:** `data/Political Violence Events by Country Mar 2026.xlsx` (ACLED: COUNTRY, MONTH, YEAR, EVENTS).
- **Optional (with `--use_extra_data`):** VIEWS Fatalities, COW Alliance v4.1 (by member yearly), Dyadic MID 4.03. Paths in `src/config.py` (`data/` folder).
- See `docs/DATA_PLAN.md` for full integration plan.

## Models

- **Target:** Next-period "high risk" (binary) or event count (regression).
- **Features:** Event lags, trend, seasonality, country dummies; with `--use_extra_data`: VIEWS fatalities, alliance count, MID count.
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

**Train**

```bash
# Base data only
python train.py --model xgb --target binary

# With VIEWS + COW alliance + MID (paths from src/config.py)
python train.py --use_extra_data --model lgbm --target binary

# Other models
python train.py --model lr --target binary
python train.py --model xgb --target regression
python train.py --model lstm --target binary --lstm_epochs 50
```

Options: `--data_path`, `--use_extra_data`, `--test_months`, `--val_months`, `--high_risk_percentile`, `--lag_months`, `--out_dir`, `--lstm_seq_len`, `--lstm_epochs`.

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
