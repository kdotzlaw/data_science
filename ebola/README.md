# Ebola Data Explorer

An interactive Streamlit dashboard that combines visual exploration of recorded Ebola outbreaks with a set of small forecasting and modeling demos. It pulls together nine CSV files (161 rows in total) covering historical outbreaks, country level yearly counts, monthly trends, synthetic clinical records, transmission risk factors, and virus reference facts.

## What it does

The app is organized into three tabs:

- **Explore** renders six interactive charts with hover context: an outbreak timeline (Gantt), case fatality rate versus outbreak size, cumulative deaths by country, an Africa bubble map of total cases and average CFR, a symptom co-occurrence heatmap, and a transmission risk factor lollipop chart. Two of the charts include country and risk category filters.
- **Methodology Demo** walks through four models: a patient outcome classifier, a CFR trend line by country, a short horizon monthly case forecast, and an outbreak severity regression. Each one shows its figure, its metrics, and a caveat.
- **About** surfaces the virus facts, virus species, and data dictionary tables so the underlying data stays reachable.

## Important caveat

The datasets are deliberately tiny (most files have fewer than 15 rows). Every model here is a **demonstration of method, not an operational prediction tool**. The honest reading is direction and relative magnitude, never precise point forecasts. The dashboard keeps this disclaimer visible so results are not mistaken for real world epidemiological guidance. Validation uses leave one out cross validation and confusion matrices rather than metrics that would look misleadingly strong at this sample size.

## Project layout

| File | Purpose |
|---|---|
| `main.py` | Streamlit entry point. Handles layout and wiring only, no data work. |
| `loaders.py` | One loader function per CSV, returning cleaned DataFrames. |
| `charts.py` | Pure chart builders. Each takes DataFrames and returns a Plotly figure. |
| `models.py` | Modeling functions. Each returns a `(metrics, figure)` pair. |
| `utils.py` | `save_figure` helper that writes each figure to `results/` as HTML and PNG. |
| `generate_results.py` | Headless script that refreshes every figure in `results/` without launching the app. |
| `test.py` | Acceptance checks for the loaders, charts, and models. |
| `PROJECT_PLAN.md` | The full step by step build plan and design notes. |
| `results/` | Saved HTML and PNG versions of every chart and model figure. |
| `*.csv` | The nine source datasets plus `data_dictionary.csv`. |

## Getting started

From the repo root:

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib plotly kaleido scikit-learn statsmodels streamlit
```

Then launch the dashboard:

```
streamlit run ebola/main.py
```

To refresh the saved figures in `results/` without opening the app:

```
python ebola/generate_results.py
```

## Stack

Python, pandas, NumPy, Plotly, scikit-learn, statsmodels, and Streamlit. Kaleido handles static PNG export of the Plotly figures.
