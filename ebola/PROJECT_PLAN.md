# Ebola Data Project Plan

A manual, step-by-step build plan for a dashboard combining **visualizations** of the
Ebola CSV files with a **forecasting / modeling** demonstration tab. Written so each
step is something you do yourself in a notebook or `main.py`; nothing here assumes
auto-generated code.

## Scope and ground rules

- **Data**: nine CSVs in this directory (see `data_dictionary.csv`).
- **Hard caveat**: row counts are small (most files <15 rows, 161 rows total). Any
  forecasting work is a **methodology demonstration**, not operational prediction.
  Build every model with this disclaimer visible in the UI.
- **Stack**: Python 3.10+, `pandas`, `matplotlib`/`plotly`, `scikit-learn`,
  `statsmodels` or `prophet`, `streamlit` (optional, for the dashboard).
- **Deliverable**: one Streamlit app (`main.py` is the entry point — currently empty)
  with two tabs: **Explore** (visualizations) and **Methodology demo** (forecasts).

## Phase 0 — Setup (30 min)

1. **Create the virtual env** (PowerShell):
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. **Install dependencies**:
   ```
   pip install pandas numpy matplotlib plotly kaleido scikit-learn statsmodels streamlit
   ```
   `kaleido` is required for Plotly's static PNG/SVG export. Add `prophet` only
   if you decide to use it in Phase 2 step 3 (heavier install).
3. **Create a `loaders.py`** in `ebola/` with one function per CSV
   (`load_country_yearly()`, `load_monthly()`, etc.). Parse dates in
   `ebola_outbreaks.csv` with `pd.to_datetime`. Return cleaned DataFrames.
4. **Create `ebola/results/`** (every chart and model figure lands here) and
   `ebola/utils.py` with a single helper used by both the dashboard and the
   headless generator script:
   ```python
   from pathlib import Path
   import plotly.graph_objects as go

   RESULTS_DIR = Path(__file__).parent / "results"
   RESULTS_DIR.mkdir(exist_ok=True)

   def save_figure(fig: go.Figure, name: str) -> None:
       """Write a Plotly figure to results/ as both interactive HTML and static PNG."""
       fig.write_html(RESULTS_DIR / f"{name}.html", include_plotlyjs="cdn")
       fig.write_image(RESULTS_DIR / f"{name}.png", width=1200, height=700, scale=2)
   ```
   Filenames are fixed per chart (see tables in Phase 1 and Phase 2) so saved
   artifacts stay stable across runs and can be diffed in git.
5. **Smoke test**: in a scratch notebook, call each loader, print `.shape` and
   `.dtypes` to confirm parsing. Then confirm `save_figure(go.Figure(), "smoke")`
   produces `results/smoke.html` and `results/smoke.png`; delete those after.

## Phase 1 — Visualizations (build in this order)

Each viz below lists **input** (which loader to call), **output** (the chart
function signature to add to `charts.py`), **steps**, **gotchas** (data quirks
specific to this chart), and an **acceptance check**. Every chart function should:

- take pre-loaded DataFrames as arguments (so `main.py` controls loading and
  caching);
- return a `plotly.graph_objects.Figure`;
- avoid `print`, `st.*`, or file-writing calls — keep them pure. **Saving is the
  caller's responsibility** (see Phase 3 step 2 and `generate_results.py`).

Each chart has a fixed filename so saved artifacts stay stable across runs:

| Chart | `name` argument to `save_figure` |
|---|---|
| 1.1 Outbreak Gantt | `outbreak_gantt` |
| 1.2 CFR vs. size | `cfr_vs_size` |
| 1.3 Cumulative deaths | `cumulative_deaths` |
| 1.4 Bubble map | `bubble_map` |
| 1.5 Symptom co-occurrence | `symptom_cooccurrence` |
| 1.6 Risk-factor lollipop | `transmission_lollipop` |

Shared imports at the top of `charts.py`:

```python
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
```

### 1.1 Outbreak Gantt chart (start here — easiest, highest payoff)

- **Input**: `loaders.load_outbreaks()` — already provides `who_emergency` (bool)
  and `duration_days`.
- **Output**: `charts.outbreak_gantt(df: pd.DataFrame) -> go.Figure`.
- **Steps**:
  1. Sort the DataFrame by `start_date` ascending so countries read chronologically
     top-to-bottom.
  2. Build the base timeline:
     ```python
     fig = px.timeline(
         df, x_start="start_date", x_end="end_date",
         y="country", color="virus_species",
         hover_data=["cases", "deaths", "fatality_rate", "description"],
     )
     fig.update_yaxes(autorange="reversed")  # earliest at the top
     ```
  3. Overlay a red diamond at the start of each WHO-emergency outbreak:
     ```python
     emerg = df[df["who_emergency"]]
     fig.add_trace(go.Scatter(
         x=emerg["start_date"], y=emerg["country"],
         mode="markers", marker_symbol="diamond",
         marker_color="red", marker_size=12,
         name="WHO emergency",
     ))
     ```
  4. Annotate the right edge of each bar with case count:
     ```python
     for _, row in df.iterrows():
         fig.add_annotation(
             x=row["end_date"], y=row["country"],
             text=f"{row['cases']:,}", showarrow=False,
             xanchor="left", xshift=4,
         )
     ```
  5. Final layout: `fig.update_layout(height=400, xaxis_title="", yaxis_title="",
     title="Recorded Ebola outbreaks")`.
- **Gotchas**: the West Africa row has country `"Guinea/Liberia/Sierra Leone"` —
  treat it as a single y category, don't try to split it.
- **Acceptance check**: 2014–2016 West Africa is the longest bar; 2026 BDBV is
  rightmost; red diamonds appear on the 2014, 2018, and 2026 rows.

### 1.2 CFR vs. outbreak size scatter

- **Input**: `loaders.load_outbreaks()`.
- **Output**: `charts.cfr_vs_size(df: pd.DataFrame) -> go.Figure`.
- **Steps**:
  1. Derive a decade label: `df = df.assign(decade=(df["start_date"].dt.year // 10
     * 10).astype(str) + "s")`.
  2. Build the scatter with log-x:
     ```python
     fig = px.scatter(
         df, x="cases", y="fatality_rate",
         size="deaths", color="decade",
         hover_name="country",
         hover_data=["virus_species", "description"],
         log_x=True,
         labels={"cases": "Cases (log)", "fatality_rate": "CFR (%)"},
     )
     ```
  3. Add an explicit log-linear fit line so the trend is visible at this small n:
     ```python
     coeffs = np.polyfit(np.log10(df["cases"]), df["fatality_rate"], 1)
     xs = np.logspace(np.log10(df["cases"].min()),
                      np.log10(df["cases"].max()), 50)
     fig.add_trace(go.Scatter(
         x=xs, y=np.polyval(coeffs, np.log10(xs)),
         mode="lines", line_dash="dot", name="log-linear fit",
     ))
     ```
  4. `fig.update_layout(title="CFR vs. outbreak size", height=450)`.
- **Gotchas**: do **not** add a confidence band — n=7 makes any band misleadingly
  tight. The dotted line is for storytelling, not inference.
- **Acceptance check**: the trend slopes downward (large outbreaks, lower CFR);
  the 1976 DRC point is top-left (small but lethal), 2014–2016 West Africa is
  bottom-right.

### 1.3 Annotated cumulative-deaths timeline

- **Input**: `loaders.load_country_yearly()` and `loaders.load_outbreak_timeline()`.
- **Output**: `charts.cumulative_deaths(yearly: pd.DataFrame, timeline:
  pd.DataFrame) -> go.Figure`.
- **Steps**:
  1. The yearly file can have multiple syndromes per country-year — aggregate first:
     ```python
     agg = yearly.groupby(["country", "year"], as_index=False)["deaths"].sum()
     agg = agg.sort_values(["country", "year"])
     agg["cumulative_deaths"] = agg.groupby("country")["deaths"].cumsum()
     ```
  2. Build the line chart:
     ```python
     fig = px.line(agg, x="year", y="cumulative_deaths",
                   color="country", markers=True)
     ```
  3. Timeline `year` includes ranges like `"2014-2016"` — collapse to a midpoint
     before annotating:
     ```python
     def _midpoint(yr):
         s = str(yr)
         if "-" in s:
             a, b = s.split("-")
             return (int(a) + int(b)) // 2
         return int(s)
     timeline = timeline.assign(plot_year=timeline["year"].apply(_midpoint))
     ```
  4. Add a faint vertical line + short annotation per timeline event:
     ```python
     for _, ev in timeline.iterrows():
         fig.add_vline(x=ev["plot_year"], line_dash="dot",
                       line_color="gray", opacity=0.4)
         fig.add_annotation(
             x=ev["plot_year"], y=1, yref="paper",
             text=(ev["notes"] or "")[:40] + "…",
             textangle=-90, showarrow=False, font_size=9,
         )
     ```
  5. `fig.update_layout(title="Cumulative deaths by country", height=500)`.
- **Gotchas**: `country_yearly` uses full country names; the West-Africa timeline
  row uses a slash-joined name that won't match any line — it appears as an
  annotation only, which is fine.
- **Acceptance check**: Sierra Leone, Liberia, and Guinea each show their largest
  step between 2014 and 2016; the line for DRC climbs in 1976, 1995, 2018, 2022.

### 1.4 Bubble map

- **Input**: `loaders.load_master()` and `loaders.load_country_yearly()`.
- **Output**: `charts.bubble_map(master: pd.DataFrame, yearly: pd.DataFrame) ->
  go.Figure`.
- **Steps**:
  1. Build a one-row-per-country coords table from yearly:
     ```python
     coords = (yearly[["iso3", "latitude", "longitude"]]
               .drop_duplicates("iso3"))
     df = master.merge(coords, on="iso3", how="left")
     ```
  2. Plot:
     ```python
     fig = px.scatter_geo(
         df, lat="latitude", lon="longitude",
         size="total_cases", color="average_cfr",
         hover_name="country",
         hover_data=["total_outbreaks", "total_deaths",
                     "latest_outbreak_year", "most_common_species"],
         color_continuous_scale="Reds",
         size_max=40, projection="natural earth",
     )
     fig.update_geos(scope="africa")
     fig.update_layout(title="Total cases and average CFR by country",
                       height=550)
     ```
- **Gotchas**: if a country in `master` is missing from `yearly` (no centroid),
  the merge produces NaN lat/lon and that bubble silently disappears. Add an
  `assert df["latitude"].notna().all()` while developing.
- **Acceptance check**: DRC, Guinea, Sierra Leone, Liberia are the four largest
  bubbles; DRC and Sudan show the deepest red (highest CFR).

### 1.5 Symptom co-occurrence heatmap

- **Input**: `loaders.load_clinical()` — `symptom_list` column already split.
- **Output**: `charts.symptom_cooccurance(df: pd.DataFrame) -> go.Figure`.
- **Steps**:
  1. Build a patient × symptom binary matrix:
     ```python
     exploded = df.explode("symptom_list")
     binary = pd.crosstab(exploded["patient_id"], exploded["symptom_list"])
     binary = (binary > 0).astype(int)
     ```
  2. Compute co-occurrence: `cooc = binary.T @ binary`. The diagonal holds each
     symptom's total frequency.
  3. Zero out the diagonal so the heatmap doesn't get dominated by it:
     `np.fill_diagonal(cooc.values, 0)`.
  4. Order symptoms by total frequency (descending) so the strongest pairs sit
     top-left:
     ```python
     order = cooc.sum().sort_values(ascending=False).index
     cooc = cooc.loc[order, order]
     ```
  5. Plot:
     ```python
     fig = px.imshow(
         cooc, text_auto=True, aspect="auto",
         color_continuous_scale="Blues",
         labels=dict(color="Co-occurrences"),
     )
     fig.update_layout(title="Symptom co-occurrence (n=10 patients)",
                       height=550)
     ```
- **Gotchas**: with only 10 patients the matrix is sparse — label the chart as
  "among reported cases", never as population prevalence. Symptoms vary in
  capitalization (`fever` vs. ` fever`) — `loaders.load_clinical()` already
  strips, but double-check if symptoms look duplicated.
- **Acceptance check**: `fever` is the highest-frequency symptom; `hemorrhage`
  and `shock` co-occur on the critical/deceased rows.

### 1.6 Risk-factor lollipop chart

- **Input**: `loaders.load_transmission_factors()` — `impact_rank` already mapped
  (Low=1 → Very High=4).
- **Output**: `charts.transmission_lollipop(df: pd.DataFrame) -> go.Figure`.
- **Steps**:
  1. Sort by category, then by score descending so categories stay grouped:
     ```python
     df = df.sort_values(["factor_category", "evidence_score"],
                        ascending=[True, False]).reset_index(drop=True)
     ```
  2. Plotly has no first-class lollipop; build it with line shapes + a scatter:
     ```python
     fig = go.Figure()
     for _, row in df.iterrows():
         fig.add_shape(
             type="line", x0=0, x1=row["evidence_score"],
             y0=row["factor"], y1=row["factor"],
             line=dict(color="lightgray", width=2),
         )
     fig.add_trace(go.Scatter(
         x=df["evidence_score"], y=df["factor"], mode="markers",
         marker=dict(
             size=14, color=df["impact_rank"],
             colorscale="Reds", showscale=True,
             colorbar=dict(title="Impact",
                           tickvals=[1, 2, 3, 4],
                           ticktext=["Low", "Med", "High", "Very high"]),
         ),
         text=df["factor_category"],
         hovertemplate="%{y}<br>Score: %{x}<br>Category: %{text}<extra></extra>",
     ))
     ```
  3. Force the y axis to match the sorted order:
     ```python
     fig.update_yaxes(categoryorder="array",
                      categoryarray=df["factor"].tolist())
     ```
  4. `fig.update_layout(xaxis_title="Evidence score (1-10)", yaxis_title="",
     title="Transmission risk factors", height=500,
     margin=dict(l=220))` — extra left margin so long factor labels fit.
- **Gotchas**: if you later add more factors with the same name across categories,
  the y axis collapses duplicates. Disambiguate by setting
  `df["factor"] = df["factor_category"] + " — " + df["factor"]`.
- **Acceptance check**: "Direct contact with body fluids", "Unsafe caregiving
  practices", and "Insufficient PPE" sit at the top-right with the darkest red
  markers; factor grouping by category is visually obvious.

## Phase 2 — Forecasting / modeling

Each model below: **input → preparation → fit → evaluate → present**. The "present"
step is the chart you add to the dashboard **and** save to `results/` via
`utils.save_figure(fig, name)`.

Put these in a new `models.py` (keep them out of `charts.py`, which stays pure
figure code). Each function takes pre-loaded DataFrames and returns a
`(metrics: dict, fig: go.Figure)` tuple — the dashboard renders `metrics` with
`st.metric`/`st.dataframe`, renders the figure with `st.plotly_chart`, then saves
the figure under the name from this table:

| Model | `name` argument to `save_figure` |
|---|---|
| 2.1 Outcome classifier coefficients | `outcome_classifier_coefs` |
| 2.2 CFR trend by country | `cfr_trend` |
| 2.3 Monthly forecast | `monthly_forecast` |
| 2.4 Severity regression | `severity_regression` |
| 2.5 Duration model | `duration_model` |

Shared imports at the top of `models.py`:

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
```

Model-specific `sklearn` / `statsmodels` imports are shown per section so you
only pull in what you use.

> **Date → string rule**: 2.3 builds hand-made `go.Scatter` traces from datetime
> values. Convert any pandas `Timestamp` to an ISO string
> (`.dt.strftime("%Y-%m-%d")`) before it touches a trace, or the PNG export via
> kaleido fails with `TypeError: Type is not JSON serializable: Timestamp`. The
> numeric-axis models (2.2, 2.4, 2.5) are unaffected.

### 2.1 Patient outcome classifier (start here — most defensible result)

- **Purpose**: Identify *which clinical factors push a patient toward death*. Fits
  a logistic regression on the 10-patient clinical records (age, incubation,
  severity, ICU, ventilation) and presents the standardized coefficients as a bar
  chart, so you can read the direction and relative strength of each risk factor.
  The deliverable is interpretability, not prediction — validated by leave-one-out
  accuracy and a confusion matrix, never AUC.
- **Input**: `loaders.load_clinical()` — `deceased` (0/1) target already present;
  `symptom_list` already split if you want symptom features.
- **Output**: `models.outcome_classifier(df: pd.DataFrame) -> tuple[dict, go.Figure]`.
- **Steps**:
  1. **Prep** — keep the feature set tiny (n=10) and **scale**, so coefficient
     magnitudes are comparable across a year-count and binary flags:
     ```python
     SEVERITY = {"Mild": 0, "Moderate": 1, "Severe": 2, "Critical": 3}
     X = pd.DataFrame({
         "age": df["age"],
         "incubation_days": df["incubation_days"],
         "severity": df["severity"].map(SEVERITY),
         "icu_admission": df["icu_admission"],
         "mechanical_ventilation": df["mechanical_ventilation"],
     })
     y = df["deceased"]
     ```
  2. **Validate with leave-one-out** (the only honest split at this n):
     ```python
     from sklearn.pipeline import make_pipeline
     from sklearn.preprocessing import StandardScaler
     from sklearn.linear_model import LogisticRegression
     from sklearn.model_selection import LeaveOneOut, cross_val_predict
     from sklearn.metrics import accuracy_score, confusion_matrix

     clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
     y_pred = cross_val_predict(clf, X, y, cv=LeaveOneOut())
     acc = accuracy_score(y, y_pred)
     cm = confusion_matrix(y, y_pred)
     ```
  3. **Refit on all rows** to read coefficients:
     ```python
     clf.fit(X, y)
     coefs = pd.Series(
         clf.named_steps["logisticregression"].coef_[0], index=X.columns
     ).sort_values()
     ```
  4. **Present** — horizontal bar chart, red = pushes toward deceased:
     ```python
     fig = go.Figure(go.Bar(
         x=coefs.values, y=coefs.index, orientation="h",
         marker_color=["crimson" if c > 0 else "steelblue" for c in coefs.values],
     ))
     fig.update_layout(
         title="Outcome classifier coefficients (positive → deceased)",
         xaxis_title="Standardized coefficient", yaxis_title="", height=400,
     )
     return {"loo_accuracy": acc, "confusion_matrix": cm.tolist()}, fig
     ```
- **Gotchas**: with 3 deceased / 7 recovered, logistic regression will nearly
  perfectly separate — coefficients are *directional*, not calibrated. Do **not**
  one-hot `exposure_type` (nearly unique per row → instant overfit). Report
  accuracy + confusion matrix, never AUC.
- **Acceptance check**: `severity`, `icu_admission`, and `mechanical_ventilation`
  are the largest positive (deceased-direction) bars.
  > **Observed mismatch (2026-06-11)**: the actual fit does *not* match this
  > expectation. Top positive bars come out `age` (+0.82) > `severity` (+0.76) >
  > `icu_admission` = `mechanical_ventilation` (+0.49 each); `incubation_days` is
  > the lone negative bar. Two reasons, both data — not a code bug: (1) `age` is a
  > clean monotonic signal (the 3 deceased are the 3 oldest) and as a granular
  > continuous feature absorbs the most standardized weight; (2) `icu_admission`
  > and `mechanical_ventilation` are **perfectly collinear** in this dataset
  > (identical columns), so logistic regression splits their shared effect in half
  > rather than ranking either near the top. LOO accuracy = 0.90, confusion matrix
  > `[[6, 1], [0, 3]]`. To make `icu`/`mech` rank as this check expects, collapse
  > the collinear pair into a single feature — a modeling change, not a fix to the
  > chart, which faithfully renders the model as specified.

### 2.2 CFR trend by country (linear with CI band)

- **Purpose**: Ask *whether a country's case fatality rate is rising or falling
  over time*. Fits a simple linear regression of CFR against year (only DRC has
  the ≥3 distinct years needed) and plots the observed points, the fitted line,
  and a confidence band — the deliberately wide band being the honest signal that
  the trend rests on very few data points.
- **Input**: `loaders.load_country_yearly()`.
- **Output**: `models.cfr_trend(yearly: pd.DataFrame) -> tuple[dict, go.Figure]`.
- **Steps**:
  1. A country-year can hold multiple syndromes — aggregate, then recompute a
     pooled CFR:
     ```python
     agg = (yearly.groupby(["country", "year"], as_index=False)
            .agg(cases=("confirmed_cases", "sum"), deaths=("deaths", "sum")))
     agg["cfr"] = agg["deaths"] / agg["cases"] * 100
     ```
  2. Fit `cfr ~ year` per country with **≥ 3 distinct years** (only DRC qualifies
     in this dataset), capturing the CI band from statsmodels:
     ```python
     import statsmodels.api as sm

     fig = go.Figure()
     slopes = {}
     for country, g in agg.groupby("country"):
         if g["year"].nunique() < 3:
             continue
         g = g.sort_values("year")
         model = sm.OLS(g["cfr"], sm.add_constant(g["year"])).fit()
         slopes[country] = round(model.params["year"], 3)
         pred = model.get_prediction(sm.add_constant(g["year"])).summary_frame()
         fig.add_trace(go.Scatter(x=g["year"], y=g["cfr"], mode="markers",
                                  name=f"{country} (obs)"))
         fig.add_trace(go.Scatter(x=g["year"], y=pred["mean"], mode="lines",
                                  name=f"{country} (fit)"))
         fig.add_trace(go.Scatter(
             x=list(g["year"]) + list(g["year"][::-1]),
             y=list(pred["mean_ci_upper"]) + list(pred["mean_ci_lower"][::-1]),
             fill="toself", fillcolor="rgba(0,0,0,0.08)",
             line_color="rgba(0,0,0,0)", showlegend=False,
         ))
     fig.update_layout(title="Case fatality rate trend", xaxis_title="Year",
                       yaxis_title="CFR (%)", height=450)
     return {"slopes": slopes}, fig
     ```
- **Gotchas**: only DRC clears the 3-year bar — say so in the UI rather than
  forcing a "trend" on 2-point countries (two points have zero residual
  degrees of freedom and an undefined CI). `year` is integer; no Timestamp
  conversion needed.
- **Acceptance check**: DRC's slope is negative; the band is visibly wide given
  only ~5 points.

### 2.3 Monthly cases short-horizon forecast

- **Purpose**: Demonstrate *near-term case forecasting* on a single country's
  monthly series. Fits simple exponential smoothing and projects 3 months ahead
  with an approximate uncertainty band. Because the series is sparse and irregular,
  the forecast is essentially flat at the last level — the point is to show
  forecasting *methodology* (and honest uncertainty), not to make an operational
  prediction.
- **Input**: `loaders.load_monthly_trends()` — `date` column already built.
- **Output**: `models.monthly_forecast(monthly: pd.DataFrame, country: str =
  "Democratic Republic of the Congo", horizon: int = 3) -> tuple[dict, go.Figure]`.
- **Steps**:
  1. Slice to one country and order by date:
     ```python
     s = monthly[monthly["country"] == country].sort_values("date")
     y = s["confirmed_cases"].astype(float).to_numpy()
     ```
  2. Fit simple exponential smoothing and forecast; build an approximate band
     from the residual spread (SES has no native interval):
     ```python
     from statsmodels.tsa.holtwinters import SimpleExpSmoothing

     fit = SimpleExpSmoothing(y, initialization_method="estimated").fit()
     fc = fit.forecast(horizon)
     resid_std = np.std(fit.resid, ddof=1) if len(y) > 1 else float(y.std() or 1)
     upper = fc + 1.96 * resid_std
     lower = np.clip(fc - 1.96 * resid_std, 0, None)
     ```
  3. Build future month labels **as strings** (kaleido-safe) and plot:
     ```python
     last = s["date"].iloc[-1]
     future = pd.date_range(last + pd.offsets.MonthBegin(1),
                            periods=horizon, freq="MS")
     obs_x = s["date"].dt.strftime("%Y-%m-%d")
     fc_x = future.strftime("%Y-%m-%d")

     fig = go.Figure()
     fig.add_trace(go.Scatter(x=obs_x, y=y, mode="lines+markers", name="observed"))
     fig.add_trace(go.Scatter(x=fc_x, y=fc, mode="lines+markers",
                              line_dash="dash", name="forecast"))
     fig.add_trace(go.Scatter(
         x=list(fc_x) + list(fc_x[::-1]),
         y=list(upper) + list(lower[::-1]),
         fill="toself", fillcolor="rgba(214,39,40,0.15)",
         line_color="rgba(0,0,0,0)", name="≈95% band",
     ))
     fig.update_layout(title=f"{country}: monthly cases + {horizon}-month forecast",
                       height=450)
     return {"forecast": fc.tolist()}, fig
     ```
- **Gotchas**: this series is **sparse and irregularly spaced** (DRC has entries
  in 2018-08, 2019-01, 2022-05/06, 2026-05) — SES essentially extrapolates the
  last level. Acceptable *only* as a methodology demo; the wide band is the
  honest part of the chart. Prophet is overkill here and won't fit < 2 cycles.
- **Acceptance check**: the forecast line is roughly flat at the last observed
  level and the shaded band is wide relative to the point forecast.

### 2.4 Outbreak severity regression

- **Purpose**: Test *how well outbreak size can be predicted from basic
  attributes* (decade, virus species, WHO-emergency status). Fits a regularized
  Ridge regression on the 7 outbreaks to predict log-scale case counts, shown as a
  predicted-vs-actual plot against a `y = x` reference line. It's a sanity demo of
  fit quality (reported as log-space MAE), confirming large epidemics land high and
  small flare-ups cluster low.
- **Input**: `loaders.load_outbreaks()` — `who_emergency` (bool) already present.
- **Output**: `models.severity_regression(outbreaks: pd.DataFrame) -> tuple[dict,
  go.Figure]`.
- **Steps**:
  1. Keep features minimal (n=7) and log-transform the heavy-tailed target:
     ```python
     df = outbreaks.copy()
     df["decade"] = df["start_date"].dt.year // 10 * 10
     df["is_ebov"] = (df["virus_species"] == "Ebola virus").astype(int)
     df["emergency"] = df["who_emergency"].astype(int)
     X = df[["decade", "is_ebov", "emergency"]].astype(float)
     y = np.log10(df["cases"])
     ```
  2. Ridge (regularized for the tiny n) with leave-one-out:
     ```python
     from sklearn.linear_model import Ridge
     from sklearn.model_selection import LeaveOneOut, cross_val_predict
     from sklearn.metrics import mean_absolute_error

     y_pred = cross_val_predict(Ridge(alpha=1.0), X, y, cv=LeaveOneOut())
     mae = mean_absolute_error(y, y_pred)
     ```
  3. **Present** — predicted-vs-actual with a `y = x` reference (numeric axes):
     ```python
     lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
     fig = go.Figure()
     fig.add_trace(go.Scatter(x=y, y=y_pred, mode="markers",
                              text=df["country"], name="outbreaks"))
     fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                              line_dash="dot", name="y = x"))
     fig.update_layout(
         title=f"Severity regression — LOO MAE = {mae:.2f} (log10 cases)",
         xaxis_title="Actual log10(cases)",
         yaxis_title="Predicted log10(cases)", height=450)
     return {"loo_mae_log10": round(mae, 3)}, fig
     ```
- **Gotchas**: 3 features / 7 rows is already near the limit — don't one-hot all
  species (that alone is 4–5 columns). Report MAE in log space; **never** R² at
  this n. The fit is a sanity demo, not a benchmark.
- **Acceptance check**: points track the diagonal loosely; the large epidemics
  (2014–2016, 2018–2020) sit at the high-cases end and the small flare-ups
  cluster low.

### 2.5 Outbreak duration model (optional / stretch)

- **Input**: `loaders.load_outbreaks()` — `duration_days` precomputed.
- **Output**: `models.duration_model(outbreaks: pd.DataFrame) -> tuple[dict,
  go.Figure]`.
- **Steps**:
  1. Drop rows with no end date, then fit a quick OLS for the headline numbers:
     ```python
     import statsmodels.api as sm

     df = outbreaks.dropna(subset=["duration_days"]).copy()
     df["year"] = df["start_date"].dt.year
     df["emergency"] = df["who_emergency"].astype(int)
     model = sm.OLS(df["duration_days"],
                    sm.add_constant(df[["year", "cases", "emergency"]])).fit()
     ```
  2. **Present** — duration vs. start year, colored by emergency status:
     ```python
     df["status"] = np.where(df["emergency"] == 1, "WHO emergency", "Contained")
     fig = go.Figure()
     for status, g in df.groupby("status"):
         fig.add_trace(go.Scatter(
             x=g["year"], y=g["duration_days"], mode="markers+text",
             text=g["country"], textposition="top center", name=status,
             marker_size=12))
     fig.update_layout(title="Outbreak duration vs. start year",
                       xaxis_title="Start year",
                       yaxis_title="Duration (days)", height=450)
     return {"coefs": model.params.round(2).to_dict()}, fig
     ```
- **Gotchas**: `duration_days` needs a parsed `end_date`; the 2026 row has one
  (`2026-05-17`) so it's included. `year` is integer — no Timestamp conversion.
- **Acceptance check**: the long-running outbreaks are the **WHO-emergency** ones
  (2014–2016 ≈ 555 days, 2018–2020 ≈ 694 days), while the contained flare-ups
  (2021, 2022) sit near ~80–110 days — i.e. severity, not recency, drives length.

## Phase 3 — Dashboard integration (2–3 hours)

By this point `loaders.py`, `charts.py`, `models.py`, and `utils.py` all exist and
are individually verified (Phases 1–2). Phase 3 is **pure wiring**: `main.py` only
loads data, lays out three tabs, and calls the existing builders. It must not
contain any data cleaning or figure-construction logic — if you find yourself
reaching for `pandas` in `main.py`, that code belongs in a loader or chart module.

Each step below lists **goal**, **steps**, **gotchas** (specific to *this*
codebase — real name mismatches and import gaps you will hit), and an
**acceptance check**.

`main.py` already contains the page config, the `tab1/tab2/tab3` split, and the
`chart()` / `model()` cache wrappers (3.3 below). Steps 3.1–3.2 reconcile its
imports; 3.4–3.6 fill the tab bodies; 3.7–3.8 add the headless generator and
verify the whole phase.

### 3.1 Reconcile `main.py` imports (do this first — two real gaps)

- **Goal**: make every builder Tab 2 needs importable, and stop relying on a
  transitive `go` import.
- **Steps**: the current header is
  ```python
  import streamlit as st
  from loaders import *
  from charts import *
  from utils import save_figure
  ```
  Add the two missing pieces:
  ```python
  import plotly.graph_objects as go     # the chart()/model() return annotations use go.*
  from models import *                  # Tab 2 calls outcome_classifier, cfr_trend, ...
  ```
- **Gotchas**:
  - **`go` is currently only in scope by accident** — `from charts import *`
    happens to re-export `go`, so the `-> go.Figure` annotations in the wrappers
    evaluate. Import `go` explicitly so a future edit to `charts.py` can't break
    `main.py`.
  - **`models` is not imported at all yet.** Without `from models import *`,
    Tab 2's `model("outcome_classifier_coefs", outcome_classifier, ...)` raises
    `NameError: outcome_classifier`. This is the first thing that will break when
    you wire Tab 2.
  - `from X import *` pulls each module's own imports (`np`, `pd`, `px`, `go`)
    into `main.py`'s namespace too. That's fine here, but it means name
    collisions are silent — keep `main.py` short so they stay visible.
- **Acceptance check**: `python -c "import main"` (or just launching Streamlit)
  imports with no `NameError` / `ImportError`.

### 3.2 Load every DataFrame once, cached

- **Goal**: one cached load per CSV, reused across reruns and across tabs, so the
  `chart()`/`model()` cache keys stay stable.
- **Steps**: right after the wrappers, load what the tabs need. Wrap the bare
  loaders so Streamlit caches the DataFrames (the loaders themselves stay
  Streamlit-free per Phase 0):
  ```python
  load = st.cache_data(show_spinner=False)

  outbreaks   = load(load_outbreaks)()
  yearly      = load(load_country_yearly)()
  timeline    = load(load_outbreak_timeline)()
  master      = load(load_master)()
  clinical    = load(load_clinical)()
  transmission = load(load_transmission_factors)()
  monthly     = load(load_monthly_trends)()
  species     = load(load_virus_species)()
  facts       = load(load_virus_facts)()
  data_dict   = load(load_data_dictionary)()
  ```
- **Gotchas**:
  - `st.cache_data` hashes the **positional args** of `chart()`/`model()`, which
    include these DataFrames. Loading them through one cached call gives each a
    stable identity, so the figure cache only busts when the underlying CSV
    actually changes — exactly the "save once per data change" behaviour Step 3.3
    promises. Re-reading a CSV inline on every rerun would defeat that.
  - Don't pre-filter here. Tab-level `st.selectbox` filtering (3.4) happens at the
    call site so the cache key reflects the filter.
- **Acceptance check**: `load_all()`-equivalent shapes print without error; the
  app's first paint reads each CSV once (subsequent reruns hit cache).

### 3.3 The `chart()` / `model()` render-and-save wrappers (already in `main.py`)

- **Goal**: render a figure **and** drop its `.html` + `.png` into `results/`
  exactly once per data change, not on every rerun.
- **Steps**: these are already written — confirm they match and understand why:
  ```python
  @st.cache_data(show_spinner=False)
  def chart(name: str, _builder, *args) -> go.Figure:
      fig = _builder(*args)
      save_figure(fig, name)
      return fig

  @st.cache_data(show_spinner=False)
  def model(name: str, _builder, *args) -> tuple[dict, go.Figure]:
      metrics, fig = _builder(*args)
      save_figure(fig, name)
      return metrics, fig
  ```
  Phase 1 builders return a bare `Figure`; Phase 2 builders return
  `(metrics, fig)` — that's why there are two wrappers.
- **Gotchas**:
  - The leading underscore in `_builder` tells `st.cache_data` **not** to hash the
    function object (unhashable) — only `name` and `*args` form the key. Keep the
    underscore.
  - Pass the `name` strings **verbatim** from the Phase 1 / Phase 2 tables; they
    are the saved filenames and the Definition-of-Done checklist greps for them.
  - `save_figure` calls kaleido for the PNG — the *first* render of each figure
    takes a couple of seconds. That cost is paid once per cache key, not per
    rerun.
- **Acceptance check**: rendering any chart twice (rerun the app) writes the PNG
  only on the first pass; the second pass is a cache hit (no spinner, no file
  mtime change).

### 3.4 Tab 1 — Explore (charts 1.1 → 1.6)

- **Goal**: six interactive charts with hover context, plus a country/year filter
  where it adds value.
- **Steps**: inside `with tab1:` render each chart through `chart(...)`, using the
  **actual** function names and the loaded frames:
  ```python
  with tab1:
      st.plotly_chart(chart("outbreak_gantt", outbreak_gantt, outbreaks),
                      use_container_width=True)
      st.plotly_chart(chart("cfr_vs_size", cfr_vs_size, outbreaks),
                      use_container_width=True)
      st.plotly_chart(chart("cumulative_deaths", cumulative_deaths, yearly, timeline),
                      use_container_width=True)
      st.plotly_chart(chart("bubble_map", bubble_map, master, yearly),
                      use_container_width=True)
      st.plotly_chart(chart("symptom_cooccurrence", symptom_cooccurance, clinical),
                      use_container_width=True)
      st.plotly_chart(chart("transmission_lollipop", transmission_lollipop, transmission),
                      use_container_width=True)
  ```
  The block above is the **baseline, no-filter render**. Two of the six charts
  benefit from a filter (`cumulative_deaths`, `transmission_lollipop`); the
  **Filters** subsection below replaces their plain calls. Leave the other four
  alone.

  **Filters (where applicable).** Add a widget only where the chart carries
  enough series/categories that hiding some *improves* readability, **and** the
  narrowed view is still meaningful. Apply that test to all six:

  | Chart | Filter on | Widget | Verdict |
  |---|---|---|---|
  | 1.3 `cumulative_deaths` | `country` | `st.multiselect` | **yes** — many country lines; isolating 1–3 is the main use |
  | 1.6 `transmission_lollipop` | `factor_category` | `st.multiselect` | **yes** — categories group cleanly; a one-category view still reads |
  | 1.2 `cfr_vs_size` | `decade` | `st.multiselect` | optional — n=7, so a filter only thins an already-sparse scatter |
  | 1.1 `outbreak_gantt` | — | — | no — the complete record *is* the point; 7 bars, nothing to hide |
  | 1.4 `bubble_map` | — | — | no — a continent overview; filtering defeats the geographic story |
  | 1.5 `symptom_cooccurance` | — | — | no — n=10 patients; any split leaves the matrix too sparse to read |

  **Keep saved artifacts on full data.** `chart(name, ...)` writes
  `results/<name>.png` for every distinct cache key. Push a *filtered* frame
  through it and the last selection silently overwrites the canonical PNG, so the
  Definition-of-Done files stop reflecting the full dataset. Split the two paths:
  render the **saved** figure with `chart()` only when the filter sits at its
  "all" default, and render every narrowed view through a **non-saving** helper:

  ```python
  @st.cache_data(show_spinner=False)
  def view(_builder, *args) -> go.Figure:
      """Filtered / interactive render — never writes to results/."""
      return _builder(*args)
  ```

  Wire `cumulative_deaths` with a country multiselect (default = every country,
  which is the path that saves the artifact):

  ```python
  countries = sorted(yearly["country"].unique())
  picked = st.multiselect("Countries", countries, default=countries,
                          key="cumdeaths_countries")
  if not picked:                                   # empty selection → nothing to plot
      st.info("Pick at least one country.")
  else:
      if set(picked) == set(countries):
          fig = chart("cumulative_deaths", cumulative_deaths, yearly, timeline)  # saves
      else:
          sub = yearly[yearly["country"].isin(picked)]
          fig = view(cumulative_deaths, sub, timeline)                          # no save
      st.plotly_chart(fig, use_container_width=True)
  ```

  Same shape for `transmission_lollipop` on `factor_category`:

  ```python
  cats = sorted(transmission["factor_category"].unique())
  picked = st.multiselect("Risk categories", cats, default=cats, key="lollipop_cats")
  if picked:
      if set(picked) == set(cats):
          fig = chart("transmission_lollipop", transmission_lollipop, transmission)
      else:
          fig = view(transmission_lollipop,
                     transmission[transmission["factor_category"].isin(picked)])
      st.plotly_chart(fig, use_container_width=True)
  ```

  For *year* ranges and one-off series hiding, prefer Plotly's built-ins — drag to
  zoom, and click a legend entry to toggle that series — over a Streamlit widget.
  Those act client-side with no rerun and never re-save the figure, so they cost
  nothing and can't touch `results/`.
  
- **Gotchas**:
  - **Function name ≠ save-`name` — still a one-word mismatch.** The function in
    `charts.py` is **`symptom_cooccurance`** (`-urance`), while the save-`name`
    filename string is `"symptom_cooccurrence"` (`-urrence`, per the Phase 1
    table). The call above pairs the real function with the filename string on
    purpose — the filename is just a label and is left unchanged so existing
    `results/` artifacts stay stable. Don't "fix" the string to match the
    function unless you also rename every `results/symptom_cooccurrence.*` file.
  - `cumulative_deaths` and `bubble_map` are the only two-arg charts — passing a
    single frame raises a `TypeError`. Mirror the signatures from `charts.py`
    exactly.
  - **Filtering through `chart()` clobbers the saved PNG.** A narrowed frame is a
    new cache key but the *same* `name`, so `save_figure` overwrites
    `results/<name>.*` with the filtered view. Route filtered renders through the
    non-saving `view()` helper (see the Filters subsection) so the canonical
    artifact always reflects full data.
- **Acceptance check**: all six charts render; hovering shows tooltips; the
  Explore tab satisfies the "at least four working charts with hover context"
  bar in the Definition of Done.

### 3.5 Tab 2 — Methodology demo (models 2.1 → 2.5)

- **Goal**: render each model's figure with its metrics surfaced and a
  per-model caveat banner.
- **Steps**: all five models now exist (Phases 2.1–2.5), so render them all.
  Factor out the repeated unpack-render-caveat into a small local helper:
  ```python
  with tab2:
      st.warning("Demonstration only — n is tiny (≤10 rows per model). "
                 "These are methodology demos, not operational predictions.")

      def demo(name, builder, *args, caption=""):
          metrics, fig = model(name, builder, *args)
          st.plotly_chart(fig, use_container_width=True)
          if caption:
              st.caption(caption)
          st.json(metrics)

      demo("outcome_classifier_coefs", outcome_classifier, clinical,
           caption="LOO accuracy + confusion matrix only — never AUC at n=10.")
      demo("cfr_trend", cfr_trend, yearly,
           caption="Only DRC clears the ≥3-distinct-years bar; band is wide on purpose.")
      demo("monthly_forecast", monthly_forecast, monthly,
           caption="SES on a sparse, irregular series — extrapolates the last level.")
      demo("severity_regression", severity_regression, outbreaks,
           caption="MAE in log10 space; a sanity demo, not a benchmark.")
      demo("duration_model", duration_model, outbreaks,
           caption="Severity (WHO-emergency), not recency, drives length.")
  ```
- **Gotchas**:
  - The original plan only promised 2.1 → 2.3 here; 2.4 and 2.5 are built now, so
    include them. The Definition of Done still only *requires* the classifier +
    one forecast, but rendering all five is free once they're wired.
  - `model()` returns `(metrics, fig)` — unpack in that order. Feeding a
    `model()` result into `st.plotly_chart` without unpacking renders nothing
    useful.
  - `st.json(metrics)` will choke on numpy types if any leak through; the model
    functions already return plain Python (`round(...)`, `.tolist()`), so keep it
    that way if you edit them.
- **Acceptance check**: each model shows a figure, a one-line caveat, and a JSON
  metrics block; the global warning banner is visible above all five.

### 3.6 Tab 3 — About (reference tables + limitations)

- **Goal**: make the data dictionary and reference CSVs reachable, with a plain
  statement of limitations.
- **Steps**:
  ```python
  with tab3:
      st.subheader("About this dataset")
      st.markdown(
          "Nine small CSVs of recorded Ebola outbreaks (161 rows total). "
          "Every model in the Methodology tab is a demonstration of method on "
          "tiny data — read directions and magnitudes, not point predictions."
      )
      st.subheader("Virus facts")
      st.dataframe(facts, use_container_width=True)
      st.subheader("Virus species")
      st.dataframe(species, use_container_width=True)
      st.subheader("Data dictionary")
      st.dataframe(data_dict, use_container_width=True)
  ```
- **Gotchas**: the Definition of Done specifically requires the **data
  dictionary** to be reachable from About — `load_data_dictionary()` exists for
  exactly this, don't skip it.
- **Acceptance check**: the three tables render and the data dictionary is
  visible from the About tab.

### 3.7 Headless generator — `generate_results.py`

- **Goal**: refresh every figure in `results/` without launching Streamlit
  (CI, or emailing a static snapshot).
- **Steps**: add `ebola/generate_results.py`. Note it imports the **real**
  function names from `charts.py`/`models.py` (so `symptom_cooccurance` — the
  function, not the `symptom_cooccurrence` filename string):
  ```python
  from loaders import (load_outbreaks, load_country_yearly,
                       load_outbreak_timeline, load_master,
                       load_clinical, load_transmission_factors,
                       load_monthly_trends)
  from charts import (outbreak_gantt, cfr_vs_size, cumulative_deaths,
                      bubble_map, symptom_cooccurance, transmission_lollipop)
  from models import (outcome_classifier, cfr_trend, monthly_forecast,
                      severity_regression, duration_model)
  from utils import save_figure

  outbreaks = load_outbreaks()
  save_figure(outbreak_gantt(outbreaks), "outbreak_gantt")
  save_figure(cfr_vs_size(outbreaks), "cfr_vs_size")

  yearly = load_country_yearly()
  timeline = load_outbreak_timeline()
  save_figure(cumulative_deaths(yearly, timeline), "cumulative_deaths")

  master = load_master()
  save_figure(bubble_map(master, yearly), "bubble_map")

  # function is spelled symptom_cooccurance; saved file is symptom_cooccurrence
  save_figure(symptom_cooccurance(load_clinical()), "symptom_cooccurrence")
  save_figure(transmission_lollipop(load_transmission_factors()),
              "transmission_lollipop")

  # Phase 2 models return (metrics, fig) — keep the figure half:
  save_figure(outcome_classifier(load_clinical())[1], "outcome_classifier_coefs")
  save_figure(cfr_trend(yearly)[1], "cfr_trend")
  save_figure(monthly_forecast(load_monthly_trends())[1], "monthly_forecast")
  save_figure(severity_regression(outbreaks)[1], "severity_regression")
  save_figure(duration_model(outbreaks)[1], "duration_model")
  ```
  Run with `python ebola/generate_results.py` (from the repo root) or
  `python generate_results.py` (from `ebola/`).
- **Gotchas**:
  - The import line is where the function-vs-filename mismatch surfaces as a hard
    `ImportError` — import `symptom_cooccurance` (the name that actually exists in
    `charts.py`), not the `symptom_cooccurrence` filename string.
  - This script must reproduce the **same** filenames as the dashboard so the two
    paths stay diff-able. Reuse the Phase 1/2 `name` strings, not ad-hoc ones.
- **Acceptance check**: a clean run writes 11 `.html` + 11 `.png` files into
  `results/` and exits 0; deleting `results/` and re-running regenerates all of
  them.

### 3.8 Run & verify the phase

- **Steps**: `streamlit run main.py`, click through all three tabs, then confirm
  `results/` holds one `.html` and one `.png` for every `name` in the Phase 1 and
  Phase 2 tables (11 figures × 2 = 22 files).
- **Acceptance check** (rolls up to the Definition of Done):
  - app opens with no traceback;
  - Explore has ≥4 hover-enabled charts;
  - Methodology demo shows the classifier + ≥1 forecast, each with a caveat;
  - the data dictionary is reachable from About;
  - `python ebola/generate_results.py` regenerates every `results/` file and
    exits cleanly.

## Phase 4 — Polish (1 hour)

- Add a download button under each chart (`st.download_button` with the underlying
  CSV slice).
- Cache loaders with `@st.cache_data`.
- Replace any hard-coded color choices with a consistent palette (one color per
  virus species across all charts).
- Add docstrings to every function in `loaders.py` and `charts.py`.

## Suggested order of work

1. Phase 0 (setup)
2. Phase 1.1, 1.2, 1.3 (gets you a usable Explore tab fast)
3. Phase 2.1 (the only model with defensible output on this data)
4. Phase 3 (wire up what exists)
5. Remaining Phase 1 and Phase 2 items as time allows
6. Phase 4 (polish only after content is in place)

## Definition of done

- `streamlit run main.py` opens without errors.
- Explore tab has at least four working charts with hover/tooltip context.
- Methodology demo tab has at least the patient outcome classifier and one
  time-series forecast, each with a visible caveat banner.
- The data dictionary is reachable from the About tab.
- `loaders.py`, `charts.py`, `models.py`, and `utils.py` exist as separate
  modules; `main.py` only does layout, not data work.
- `ebola/results/` contains an `.html` and a `.png` for every filename listed in
  the Phase 1 and Phase 2 tables.
- `python ebola/generate_results.py` regenerates every file in `results/`
  without launching Streamlit, and exits cleanly.
