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

## Verification
1. Run `python Covid-Test/covid.py`
2. Confirm the choropleth opens in browser with animation controls
3. Verify early frames (Jan 2020) show cases concentrated in China
4. Verify later frames (July 2020) show global spread with US/Brazil/India prominent
5. Hover over countries to confirm raw case numbers display correctly
