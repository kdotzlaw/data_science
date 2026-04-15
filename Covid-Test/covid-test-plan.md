# Plan: Animated COVID-19 Spread Visualization

## Context
The user wants to visualize how COVID-19 spread through countries over time using `covid_19_clean_complete.csv`. The dataset contains 49,068 rows spanning 2020-01-22 to 2020-07-27 with 187 countries, including Lat/Long, confirmed/deaths/recovered/active counts, and WHO Region. The target file is `Covid-Test/covid.py` (currently empty).

## Approach
Build an animated choropleth world map using `plotly.express.choropleth` with weekly date frames. Plotly is already installed (v6.0.1) alongside pandas.

## Files
- **Modify:** `Covid-Test/covid.py` (write the full script)
- **Read:** `Covid-Test/covid_19_clean_complete.csv` (data source)

## Implementation Steps

### 1. Imports & Data Loading

**Imports required:**
- `pandas` (v2.2.3) — data loading, groupby, date parsing
- `numpy` (installed) — `log10` and `clip` for scale transformation
- `plotly.express` (v6.0.1) — `choropleth()` with built-in `animation_frame` support

**Data loading details:**
- File path: `covid_19_clean_complete.csv` (relative, same directory as `covid.py`)
- Use `pd.read_csv()` with `parse_dates=['Date']` to auto-parse the `Date` column to `datetime64`
- The `Date` column is stored as string format `YYYY-MM-DD` (e.g., `2020-01-22`), which pandas parses natively
- No encoding issues expected — file is standard UTF-8 CSV
- Result: DataFrame with 49,068 rows, 10 columns
- Column dtypes after loading: `Province/State` (object, 34,404 nulls), `Country/Region` (object), `Lat` (float64), `Long` (float64), `Date` (datetime64), `Confirmed`/`Deaths`/`Recovered`/`Active` (int64), `WHO Region` (object)

---

### 2. Aggregate to Country Level

**Why this is needed:**
8 countries have province/state-level rows that must be combined:
| Country | Provinces |
|---|---|
| China | 33 |
| Canada | 12 |
| France | 10 |
| United Kingdom | 10 |
| Australia | 8 |
| Netherlands | 3 |
| Denmark | 1 |
| Greenland | 1 |

Without aggregation, these countries would show only partial data (one province per row), and `px.choropleth` would use whichever row it encounters last for a given country+date.

**Aggregation strategy:**
- Group by: `['Country/Region', 'Date']`
- Aggregation functions per column:
  - `Confirmed`: `sum` — total confirmed cases across all provinces
  - `Deaths`: `sum`
  - `Recovered`: `sum`
  - `Active`: `sum`
  - `Lat`: `mean` — average latitude across provinces (used only if bubble map is added later; choropleth doesn't use it)
  - `Long`: `mean` — average longitude
  - `WHO Region`: `first` — constant per country, just take the first value
- Use `.agg()` with a dict to specify per-column functions
- Call `.reset_index()` to flatten the grouped DataFrame back to columns

**Expected result:** ~187 countries x 188 dates = ~35,156 rows (matches the `full_grouped.csv` row count, which is a good sanity check)

---

### 3. Country Name Mapping

**Why this is needed:**
`px.choropleth` with `locationmode='country names'` matches against plotly's internal list of country names (based on the Natural Earth dataset). Several names in the CSV differ from what plotly expects. Unmatched countries silently fail to render — no error, just missing from the map.

**Complete mapping dict** (all mismatches from the 187 unique country names in the dataset):
```python
COUNTRY_NAME_MAP = {
    'Burma': 'Myanmar',
    'Cabo Verde': 'Cape Verde',
    'Congo (Brazzaville)': 'Republic of the Congo',
    'Congo (Kinshasa)': 'Democratic Republic of the Congo',
    'Cote d\'Ivoire': 'Ivory Coast',
    'Czechia': 'Czech Republic',
    'Eswatini': 'Eswatini',          # plotly 6.x accepts this; verify
    'Holy See': 'Vatican City',       # too small to render but map anyway
    'Korea, South': 'South Korea',    # already 'South Korea' in data — defensive
    'North Macedonia': 'North Macedonia',  # plotly 6.x accepts; verify
    'Taiwan*': 'Taiwan',
    'US': 'United States of America',
    'West Bank and Gaza': 'Palestine',
}
```

**Application:** `df['Country/Region'] = df['Country/Region'].replace(COUNTRY_NAME_MAP)`

**Countries that will not render regardless** (too small for Natural Earth polygons):
- Holy See / Vatican City
- Monaco
- Liechtenstein
- San Marino
- These are acceptable omissions — invisible at world scale

**Validation approach:** After building the figure, compare `df['Country/Region'].nunique()` against the number of rendered locations to identify any remaining mismatches.

---

### 4. Downsample to Weekly Dates

**Why this is needed:**
- 188 daily animation frames create a sluggish, hard-to-control animation
- Weekly sampling gives ~27 frames — smooth enough to show progression, fast enough to be interactive
- The slider/scrubber remains usable with 27 stops

**Implementation details:**
1. Get sorted list of unique dates: `dates = sorted(df['Date'].unique())`
2. Select every 7th date: `weekly_dates = dates[::7]`
3. Always include the final date (2020-07-27) if it wasn't already selected:
   ```python
   if dates[-1] not in weekly_dates:
       weekly_dates.append(dates[-1])
   ```
4. Filter the DataFrame: `df = df[df['Date'].isin(weekly_dates)]`
5. Add string column for animation frame labels:
   ```python
   df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
   ```
   This is needed because `animation_frame` displays the value as-is — `datetime64` objects would show timestamps.

**Expected result:** ~27 frames x 187 countries = ~5,049 rows — well within plotly's performance envelope.

---

### 5. Log-Scale Column

**Why this is needed:**
- On 2020-07-27, confirmed cases range from 0 (a few countries) to 4,290,259 (US)
- On a linear color scale, only the top 3-5 countries would show any color; everything else would appear as the minimum
- Log10 compresses this to a 0–6.63 range, making differences visible across all magnitudes

**Implementation details:**
1. Clip confirmed values to a minimum of 1 to avoid `log10(0)` which is `-inf`:
   ```python
   df['Confirmed_log'] = np.log10(df['Confirmed'].clip(lower=1))
   ```
2. This maps values as follows:
   - 0 cases → `log10(1)` = 0.0 (lightest color)
   - 10 cases → 1.0
   - 100 cases → 2.0
   - 1,000 cases → 3.0
   - 10,000 cases → 4.0
   - 100,000 cases → 5.0
   - 1,000,000 cases → 6.0
   - 4,290,259 cases → 6.63 (darkest color)
3. Countries with 0 confirmed cases will map to 0.0 (the same as 1 case) — this is acceptable since the lightest color effectively reads as "no/minimal cases"

---

### 6. Build Animated Choropleth

**Function call:** `plotly.express.choropleth()`

**Parameters in detail:**
| Parameter | Value | Reason |
|---|---|---|
| `data_frame` | `df` | The aggregated, mapped, weekly DataFrame |
| `locations` | `'Country/Region'` | Column with country names |
| `locationmode` | `'country names'` | Match by name (not ISO codes) |
| `color` | `'Confirmed_log'` | Log-scaled confirmed cases for color intensity |
| `animation_frame` | `'Date_str'` | Each unique date string becomes one animation frame |
| `color_continuous_scale` | `'YlOrRd'` | Yellow → Orange → Red sequential palette; intuitive "heat" metaphor for severity |
| `range_color` | `[0, df['Confirmed_log'].max()]` | **Fixed range across all frames** — prevents the color scale from auto-adjusting per frame, which would make early frames misleadingly intense |
| `hover_name` | `'Country/Region'` | Country name as hover title |
| `hover_data` | `{'Confirmed': ':,', 'Deaths': ':,', 'Recovered': ':,', 'Confirmed_log': False}` | Show raw counts with comma formatting on hover; hide the log column |
| `title` | `'COVID-19 Confirmed Cases by Country (Jan–Jul 2020)'` | Descriptive title |
| `labels` | `{'Confirmed_log': 'Confirmed Cases'}` | Clean label for the color axis |

**How animation_frame works internally:**
- Plotly creates one `choropleth` trace per frame
- The slider and play/pause button are auto-generated
- Frame transitions are instant (no interpolation between choropleth frames)

---

### 7. Layout Customization

**Geo projection settings** via `fig.update_layout(geo=...)`:
```python
geo=dict(
    showframe=False,          # no border around the map
    showcoastlines=True,      # keep coastlines for geographic reference
    projection_type='natural earth',  # familiar flat map projection
)
```

**Colorbar customization** via `fig.update_layout(coloraxis_colorbar=...)`:
- Override tick positions and labels to show human-readable case counts instead of log values:
```python
coloraxis_colorbar=dict(
    title='Confirmed Cases',
    tickvals=[0, 1, 2, 3, 4, 5, 6],
    ticktext=['1', '10', '100', '1K', '10K', '100K', '1M'],
)
```
- This directly maps the log10 scale back to intuitive numbers

**Slider label** via `fig.update_layout(sliders=...)`:
```python
sliders=[dict(currentvalue=dict(prefix='Date: '))]
```
- Adds "Date: " prefix to the currently-selected frame label shown above the slider

**Figure dimensions:**
```python
fig.update_layout(height=600, width=1000)
```
- 1000x600 fits comfortably in a browser window while giving the map enough horizontal space for the Natural Earth projection

---

### 8. Display

**Primary display:** `fig.show()`
- Opens the interactive figure in the system's default web browser
- Renders via plotly's JavaScript library (plotly.js) in an auto-generated HTML page
- The user gets play/pause controls, a date slider, hover tooltips, zoom/pan, and a screenshot button

**Optional HTML export:** `fig.write_html('covid_choropleth.html')`
- Saves a standalone HTML file that can be shared or reopened without Python
- Consider adding this as a commented-out line for convenience

## Verification (Steps 1–8)
1. Run `python Covid-Test/covid.py`
2. Confirm the choropleth opens in browser with animation controls
3. Verify early frames (Jan 2020) show cases concentrated in China
4. Verify later frames (July 2020) show global spread with US/Brazil/India prominent
5. Hover over countries to confirm raw case numbers display correctly

---
---

# Expansion Plan

## Context
The base choropleth (`covid.py` → `covid_choropleth.html`) is complete. The following covers 5 major additions: new visualizations on existing data, worldometer analysis, USA county drill-down, forecasting, and an interactive Dash dashboard.

## Prerequisites
- `pip install statsmodels` (for forecasting)
- `pip install dash` (for dashboard)

## New File Structure
```
Covid-Test/
  covid.py                     # existing — no changes
  shared_utils.py              # NEW — shared constants and helpers
  covid_bubble_map.py          # Addition 1a
  covid_mortality_rate.py      # Addition 1b
  covid_recovery_rate.py       # Addition 1c
  covid_who_regions.py         # Addition 1d
  covid_worldometer.py         # Addition 2
  covid_usa_counties.py        # Addition 3
  covid_forecast.py            # Addition 4
  covid_dashboard.py           # Addition 5
```

## Implementation Order
1. `shared_utils.py` first (all scripts depend on it)
2. Addition 1 (1a–1d) — simplest, validates shared utilities
3. Addition 2 — exercises worldometer name mapping
4. Addition 3 — standalone, large data
5. Addition 4 — requires statsmodels install
6. Addition 5 — integrates everything, do last

---

## Step 0: Shared Utilities (`shared_utils.py`)

Extract reusable pieces from `covid.py` into a shared module that all new scripts import.

**Contents:**

1. `COUNTRY_NAME_MAP` — the existing 13-entry dict from `covid.py:20-34`

2. `WORLDOMETER_TO_CLEAN_MAP` — new dict mapping worldometer country names to `covid_19_clean_complete.csv` names:
```python
WORLDOMETER_TO_CLEAN_MAP = {
    'USA': 'US',
    'UK': 'United Kingdom',
    'UAE': 'United Arab Emirates',
    'S. Korea': 'South Korea',
    'CAR': 'Central African Republic',
    'Congo': 'Congo (Brazzaville)',
    'DRC': 'Congo (Kinshasa)',
    'Ivory Coast': "Cote d'Ivoire",
    'Myanmar': 'Burma',
    'Palestine': 'West Bank and Gaza',
    'St. Vincent Grenadines': 'Saint Vincent and the Grenadines',
    'Taiwan': 'Taiwan*',
    'Vatican City': 'Holy See',
}
```
Apply `WORLDOMETER_TO_CLEAN_MAP` first (to raw names), then `COUNTRY_NAME_MAP` (for plotly).

3. `PLOTLY_GEO_LAYOUT` — the geo dict: `dict(showframe=False, showcoastlines=True, projection_type='natural earth')`

4. `load_and_aggregate()` — reads `covid_19_clean_complete.csv`, groupby country+date, agg sum/mean/first. Returns DataFrame.

5. `downsample_weekly(df)` — takes aggregated df, selects every 7th date + final date, adds `Date_str`. Returns filtered df.

---

## Addition 1: Additional Visualizations

### 1a. Bubble Map (`covid_bubble_map.py` → `covid_bubble_map.html`)

**Data:** `covid_19_clean_complete.csv` via `load_and_aggregate()` + `downsample_weekly()`

**Steps:**
1. Load, aggregate, apply `COUNTRY_NAME_MAP`, downsample weekly
2. Add `Confirmed_sqrt = np.sqrt(df['Confirmed'].clip(lower=0))` — sqrt scaling is better than log for bubble area perception
3. Build figure:
   - `px.scatter_geo()` with `lat='Lat'`, `lon='Long'`, `size='Confirmed_sqrt'`, `color='WHO Region'`
   - `animation_frame='Date_str'`, `size_max=40`, `projection='natural earth'`
   - `hover_data`: raw Confirmed/Deaths with `:,` formatting; hide Confirmed_sqrt/Lat/Long
4. Apply geo layout, slider prefix `'Date: '`, size 1000x600
5. `fig.write_html('covid_bubble_map.html', auto_open=True)`

### 1b. Mortality Rate Choropleth (`covid_mortality_rate.py` → `covid_mortality_rate.html`)

**Data:** Same as 1a

**Steps:**
1. Load, aggregate, name-map, downsample
2. Compute mortality rate with zero-division guard:
   ```python
   df['Mortality_Rate'] = np.where(df['Confirmed'] > 0, df['Deaths'] / df['Confirmed'] * 100, 0.0)
   ```
3. Build `px.choropleth()`:
   - `color='Mortality_Rate'`, `color_continuous_scale='Reds'`
   - `range_color=[0, 15]` — cap at 15% to prevent small-sample outliers from washing out the scale; true values still visible in hover
   - `hover_data`: Deaths, Confirmed (`:,`), Mortality_Rate (`:.2f`)
4. Colorbar title: `'Mortality Rate (%)'`
5. Write HTML

### 1c. Recovery Rate Choropleth (`covid_recovery_rate.py` → `covid_recovery_rate.html`)

**Data:** Same as 1a

**Steps:**
1. Load, aggregate, name-map, downsample
2. Compute:
   ```python
   df['Recovery_Rate'] = np.where(df['Confirmed'] > 0, df['Recovered'] / df['Confirmed'] * 100, 0.0)
   ```
3. Build `px.choropleth()`:
   - `color='Recovery_Rate'`, `color_continuous_scale='Greens'`, `range_color=[0, 100]`
   - Add note in title: `'... — Data may be incomplete'` (US stopped reporting recovered in some periods)
4. Write HTML

### 1d. WHO Region Comparison (`covid_who_regions.py` → `covid_who_regions.html`)

**Data:** `full_grouped.csv` (35,156 rows, already country-level, has `WHO Region` and `New cases` columns)

**Steps:**
1. Load: `pd.read_csv('full_grouped.csv', parse_dates=['Date'])`
2. Group by `['Date', 'WHO Region']`, sum `Confirmed`, `Deaths`, `Recovered`, `New cases`
3. Build line chart:
   - `px.line()` with `x='Date'`, `y='Confirmed'`, `color='WHO Region'`
   - `hover_data={'Confirmed': ':,'}`
   - Title: `'COVID-19 Total Confirmed Cases by WHO Region'`
4. Write HTML

---

## Addition 2: Worldometer Analysis (`covid_worldometer.py` → `covid_worldometer.html`)

**Data:** `worldometer_data.csv` (209 rows, snapshot)

### 2a. Cases Per Million (static choropleth)

**Steps:**
1. Load: `pd.read_csv('worldometer_data.csv')`
2. Drop rows with null Population: `wm.dropna(subset=['Population'])`
3. Apply `WORLDOMETER_TO_CLEAN_MAP` then `COUNTRY_NAME_MAP` to `Country/Region`
4. Compute: `wm['Cases_Per_Million'] = wm['TotalCases'] / wm['Population'] * 1e6`
5. Log-scale: `wm['CPM_log'] = np.log10(wm['Cases_Per_Million'].clip(lower=1))`
6. Build static `px.choropleth()` (no `animation_frame`):
   - `color='CPM_log'`, `color_continuous_scale='Viridis'`
   - `hover_data`: raw Cases_Per_Million (`:.0f`), TotalCases, Population (`:,`); hide CPM_log
   - Custom colorbar ticks: `[0,1,2,3,4]` → `['1','10','100','1K','10K']`
7. Write HTML

### 2b. Healthcare Burden (horizontal bar chart)

**Steps:**
1. From same worldometer df, drop rows with null `Serious,Critical`: keeps ~122 countries
2. Compute: `burden['Critical_Rate'] = burden['Serious,Critical'] / burden['TotalCases'] * 100`
3. Take top 30 by `Critical_Rate` for readability
4. Build `px.bar()`:
   - `x='Critical_Rate'`, `y='Country/Region'`, `orientation='h'`, `color='Continent'`
   - `yaxis={'categoryorder': 'total ascending'}`, `height=800`
   - `hover_data`: raw Serious,Critical and TotalCases (`:,`)
5. Write HTML (separate file `covid_healthcare_burden.html` or combined)

---

## Addition 3: USA County Drill-Down (`covid_usa_counties.py` → `covid_usa_counties.html`)

**Data:** `usa_county_wise.csv` (627,920 rows, daily, 1,978 counties)

**Steps:**
1. Download and cache US counties GeoJSON (~17 MB):
   - URL: `https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json`
   - Save locally as `us_counties_geojson.json`; skip download if file already exists
2. Load: `pd.read_csv('usa_county_wise.csv')`
3. Parse dates: `pd.to_datetime(df['Date'], format='%m/%d/%y')` — note MM/DD/YY format
4. Drop null FIPS (1,880 rows — aggregate entries like "Kansas City", correctional facilities)
5. Format FIPS as 5-digit zero-padded string: `df['FIPS'].astype(int).astype(str).str.zfill(5)`
6. Downsample to weekly (~25 frames)
7. Log-scale: `np.log10(df['Confirmed'].clip(lower=1))`
8. Build `px.choropleth()`:
   - `geojson=counties`, `locations='FIPS'`, `color='Confirmed_log'`
   - `animation_frame='Date_str'`, `scope='usa'`
   - `color_continuous_scale='YlOrRd'`, fixed `range_color`
   - `hover_name='Admin2'` (county name), show Province_State, Confirmed, Deaths
9. Layout: `scope='usa'`, `showlakes=True`, colorbar ticks `[0–5]` → `['1'–'100K']`
10. Write HTML

**Performance note:** ~1,978 counties x ~25 frames + 17MB GeoJSON = large output (50–100 MB). Consider biweekly frames (`dates[::14]`) if too slow.

---

## Addition 4: Forecasting (`covid_forecast.py` → `covid_forecast.html`)

**Data:** `day_wise.csv` (188 rows, daily global aggregates with Confirmed/Deaths/Recovered columns)

**Dependency:** `pip install statsmodels`

**Steps:**
1. Load: `pd.read_csv('day_wise.csv', parse_dates=['Date'])`
2. Fit Holt-Winters exponential smoothing:
   ```python
   from statsmodels.tsa.holtwinters import ExponentialSmoothing
   model = ExponentialSmoothing(df['Confirmed'], trend='add', seasonal=None, initialization_method='estimated')
   fit = model.fit()
   ```
3. Forecast 30 days beyond 2020-07-27:
   ```python
   forecast = fit.forecast(steps=30)
   ```
4. Build confidence intervals from residual std:
   ```python
   std_resid = fit.resid.std()
   upper = forecast + 1.96 * std_resid
   lower = forecast - 1.96 * std_resid
   ```
5. Build figure with `plotly.graph_objects`:
   - `go.Scatter` for historical line (solid)
   - `go.Scatter` for forecast line (dashed)
   - `go.Scatter` filled area for 95% CI band
6. Write HTML

**Fallback (if statsmodels unavailable):** Use `scipy.optimize.curve_fit` with a logistic growth model `L / (1 + exp(-k*(t-t0)))`, initial guess `p0=[2e7, 0.05, 100]`. Less robust but requires no additional packages.

---

## Addition 5: Dashboard (`covid_dashboard.py` → local server at `http://127.0.0.1:8050`)

**Dependency:** `pip install dash`

**Components:**
1. **Global choropleth** with a date slider — single-date snapshot, updates on slider change
2. **Country dropdown** (multi-select) — defaults to `'United States of America'`
3. **Metric radio buttons** — Confirmed / Deaths / Recovered / Active
4. **Time-series line chart** — updates when country selection or metric changes

**Structure:**
1. Import from `shared_utils`: `load_and_aggregate`, `COUNTRY_NAME_MAP`, `PLOTLY_GEO_LAYOUT`
2. Load data once at module level (no weekly downsampling — dashboard uses daily data with slider)
3. `app.layout`: `html.Div` with `dcc.Graph` (choropleth), `dcc.Slider` (dates), `dcc.Dropdown` (countries), `dcc.RadioItems` (metric), `dcc.Graph` (time-series)
4. Two callbacks:
   - `update_choropleth(date_idx, metric)` — filters to selected date, log-scales metric, returns `px.choropleth()`
   - `update_timeseries(selected_countries, metric)` — filters to selected countries, returns `px.line()`
5. `app.run(debug=True)` — Dash 2.x API; runs on Flask (already installed)

**Slider marks:** Show label every 14th date to avoid overcrowding: `{i: d.strftime('%b %d') for i, d in enumerate(dates) if i % 14 == 0}`

---

## Verification (Expansion)

| Addition | How to verify |
|----------|---------------|
| shared_utils | `python -c "from shared_utils import load_and_aggregate; print(load_and_aggregate().shape)"` → `(35156, 8)` |
| 1a Bubble map | Open `covid_bubble_map.html`; bubbles grow over time, colored by WHO Region |
| 1b Mortality | Open HTML; early frames show high rates in small-sample countries; cap at 15% prevents washout |
| 1c Recovery | Open HTML; green gradient, rates increase over time for most countries |
| 1d WHO Regions | Open HTML; 6 lines diverge, Americas/Europe dominate by July |
| 2a Cases/million | Open HTML; small countries (Qatar, Bahrain) rank higher than US/Brazil |
| 2b Healthcare | Open HTML; horizontal bar chart, top 30 countries by critical rate |
| 3 USA counties | Open HTML; early frames show WA/NY, spread fills in over time; check FIPS rendering |
| 4 Forecasting | Open HTML; dashed forecast line extends 30 days past July 27 with CI band |
| 5 Dashboard | Run `python covid_dashboard.py`, open `http://127.0.0.1:8050`; slider/dropdown/radio all update charts |
