# Data integration plan

## Sources and use

| Source | File(s) | Use |
|--------|--------|-----|
| **Main panel** | `data/Political Violence Events by Country Mar 2026.xlsx` | Base: country, month, year, events |
| **VIEWS Fatalities** | `data/VIEWS Fatalities Data 2026.csv` | Features: main_mean_ln, main_mean, main_dich (country-month) |
| **COW Alliance v4.1** | `data/version4.1_csv/alliance_v4.1_by_member_yearly.csv` | Feature: alliance count per country-year |
| **Dyadic MID 4.03** | `data/dyadic_mid_4.03.csv` | Feature: MID count per country-year (statea + stateb) |
| **ATOP S-score** | `data/Atop Sscore.csv` | Optional: foreign-policy similarity (dyad-year → country-year aggregate) |
| **Regional ACLED** | Africa, Asia Pacific, Europe, LAC, Middle East, US/Canada Excel | Optional: Admin1/week spillover or regional totals |
| **GED / MIDLOC** | Event-level; heavy | Optional later: geographic spillover |

## Pipeline

1. **Build base panel** from Political Violence Excel (country, year, month, period_index, events).
2. **Merge VIEWS** on (country, year, month); fill missing with 0 or median.
3. **Merge country-year** features: alliance count, MID count (from COW/state codes; need country-name ↔ ccode map for alignment).
4. **Features**: lags, trend, rolling, month dummies, country dummies, + views_fatalities + alliance_count + mid_count.
5. **Model**: LR/XGB/LSTM as now; add LGBM with Tweedie for count regression.

## Country alignment

- Political Violence & VIEWS: same country names for 179 countries; left-join VIEWS to panel.
- COW alliance/MID use ccode (statea, stateb). We use state_name from alliance file to align to panel country names where possible; else skip or map via gwcode/iso.
