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
- **Output**: `charts.symptom_cooccurrence(df: pd.DataFrame) -> go.Figure`.
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

| Model | `name` argument to `save_figure` |
|---|---|
| 2.1 Outcome classifier coefficients | `outcome_classifier_coefs` |
| 2.2 CFR trend by country | `cfr_trend` |
| 2.3 Monthly forecast | `monthly_forecast` |
| 2.4 Severity regression | `severity_regression` |
| 2.5 Duration model | `duration_model` |

### 2.1 Patient outcome classifier (start here — most defensible result)

- **Input**: `ebola_clinical.csv` (10 rows — only useful as a teaching example)
- **Steps**:
  1. **Prep**: encode `sex` (M/F → 0/1), `severity` ordinal
     (`Mild<Moderate<Severe<Critical`), one-hot `exposure_type`, drop free-text
     `symptoms` (or split into binary indicators for top 5 symptoms).
  2. **Target**: `outcome` → binary (`Recovered=0, Deceased=1`).
  3. **Model**: `sklearn.linear_model.LogisticRegression(max_iter=1000)`.
  4. **Validate**: `LeaveOneOut` CV; report mean accuracy and a 2×2 confusion
     matrix. Do **not** report AUC on n=10.
  5. **Present**: bar chart of model coefficients (which features push toward
     deceased vs. recovered).
- **Acceptance check**: `severity=Critical`, `icu_admission`, and
  `mechanical_ventilation` come out as the strongest deceased-direction features.

### 2.2 CFR trend by country (linear / LOESS)

- **Input**: `ebola_country_yearly.csv`
- **Steps**:
  1. For each country with ≥ 3 rows, fit `case_fatality_rate ~ year` using
     `numpy.polyfit(deg=1)` or `statsmodels.OLS`.
  2. Plot the observed points and the fit line per country (small multiples).
  3. Report slope and a wide confidence band (`statsmodels` gives this for free).
- **Acceptance check**: DRC shows a downward CFR slope; you can articulate this as
  "response capability improving" with an honest "n is small" caveat.

### 2.3 Monthly cases short-horizon forecast

- **Input**: `ebola_monthly_trends.csv` (10 rows total — illustrative only)
- **Steps**:
  1. Pick **one country** with the most rows (DRC has the most monthly entries).
  2. Build a date index: `pd.to_datetime(df.year.astype(str) + "-" +
     df.month.astype(str) + "-01")`.
  3. **Model A — exponential smoothing**:
     `statsmodels.tsa.holtwinters.SimpleExpSmoothing(series).fit()`.
  4. **Model B — Prophet** (optional): only if you want to demonstrate uncertainty
     bands. Set `interval_width=0.95` so the bands look appropriately wide.
  5. **Present**: line chart of observed + 3-month forecast with confidence band.
- **Acceptance check**: the band is wide enough that the user reading it immediately
  understands "this is illustrative." If the band looks tight, the chart is lying.

### 2.4 Outbreak severity regression

- **Input**: `ebola_outbreaks.csv` (7 rows)
- **Steps**:
  1. Features: decade (from `start_date`), `virus_species` (one-hot),
     `who_emergency_status` (binary), country group.
  2. Targets: `cases` (log-transformed) and `deaths` (log-transformed) — fit
     separately.
  3. Model: `sklearn.linear_model.Ridge(alpha=1.0)` (regularize given n=7).
  4. **Validate**: leave-one-out; report MAE in log space. Do **not** report R².
  5. **Present**: predicted vs. actual scatter with the diagonal `y=x` overlaid.
- **Acceptance check**: WHO-emergency outbreaks predict to larger case counts; the
  scatter is a sanity demo, not a benchmark.

### 2.5 Outbreak duration model (optional / stretch)

- **Input**: `ebola_outbreaks.csv`
- **Steps**:
  1. Compute `duration_days = (end_date - start_date).dt.days`.
  2. Regress `duration_days ~ year + cases + who_emergency_status`.
  3. Plot duration vs. start year, colored by emergency status.
- **Acceptance check**: post-2015 outbreaks have shorter durations on average (this
  is the story — improved response time).

## Phase 3 — Dashboard integration (2–3 hours)

1. **Structure** `main.py`:
   ```python
   import streamlit as st
   from loaders import *
   from charts import *
   from utils import save_figure

   st.set_page_config(page_title="Ebola Data Explorer", layout="wide")
   tab1, tab2, tab3 = st.tabs(["Explore", "Methodology demo", "About"])
   ```
2. **Render and save in one step**. Wrap each figure builder so the
   `save_figure` call runs once per data change instead of on every Streamlit
   rerun:
   ```python
   @st.cache_data(show_spinner=False)
   def build_and_save(name: str, _builder, *args) -> go.Figure:
       fig = _builder(*args)
       save_figure(fig, name)
       return fig

   fig = build_and_save("outbreak_gantt", outbreak_gantt, outbreaks_df)
   st.plotly_chart(fig, use_container_width=True)
   ```
   Use the `name` values from the Phase 1 and Phase 2 tables verbatim.
3. **Tab 1 (Explore)**: render charts 1.1 → 1.6 via `build_and_save`. Put
   `st.selectbox` filters for country/year above the relevant charts.
4. **Tab 2 (Methodology demo)**: render models 2.1 → 2.3 via `build_and_save`.
   Every model gets a
   `st.warning("Demonstration only — n is too small for operational use.")` banner.
5. **Tab 3 (About)**: render `ebola_virus_facts.csv` and `ebola_virus_species.csv`
   as reference tables, plus a paragraph describing data sources and limitations.
6. **Run locally**: `streamlit run main.py`. After the first load, confirm
   `results/` contains one `.html` and one `.png` per filename in the tables.
7. **Headless batch alternative** — add `ebola/generate_results.py` so figures
   can be refreshed without launching Streamlit (useful for CI or for emailing a
   static snapshot):
   ```python
   from loaders import (load_outbreaks, load_country_yearly,
                        load_outbreak_timeline, load_master,
                        load_clinical, load_transmission_factors,
                        load_monthly_trends)
   from charts import (outbreak_gantt, cfr_vs_size, cumulative_deaths,
                       bubble_map, symptom_cooccurrence, transmission_lollipop)
   from utils import save_figure

   outbreaks = load_outbreaks()
   save_figure(outbreak_gantt(outbreaks), "outbreak_gantt")
   save_figure(cfr_vs_size(outbreaks), "cfr_vs_size")

   yearly = load_country_yearly()
   timeline = load_outbreak_timeline()
   save_figure(cumulative_deaths(yearly, timeline), "cumulative_deaths")

   master = load_master()
   save_figure(bubble_map(master, yearly), "bubble_map")

   save_figure(symptom_cooccurrence(load_clinical()), "symptom_cooccurrence")
   save_figure(transmission_lollipop(load_transmission_factors()),
               "transmission_lollipop")

   # Append Phase 2 model figures here as you build them, using the names
   # from the Phase 2 table.
   ```
   Run with `python ebola/generate_results.py`.

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
- `loaders.py`, `charts.py`, and `utils.py` exist as separate modules;
  `main.py` only does layout, not data work.
- `ebola/results/` contains an `.html` and a `.png` for every filename listed in
  the Phase 1 and Phase 2 tables.
- `python ebola/generate_results.py` regenerates every file in `results/`
  without launching Streamlit, and exits cleanly.
