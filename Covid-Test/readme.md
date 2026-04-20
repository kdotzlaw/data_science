# COVID-19 Data Exploration

An interactive exploration of the 2020 Johns Hopkins / Worldometer COVID-19 datasets. Produces a suite of standalone animated HTML visualizations plus a Dash dashboard that ties them together.

## What's here

All charts are driven by the CSVs in [data/](data/) and share loading + country-name-mapping logic via [shared_utils.py](shared_utils.py). Each script in [src/](src/) writes its output HTML into [result/](result/).

### Animated world choropleth: [src/covid.py](src/covid.py)
Weekly-frame animated choropleth of global confirmed cases (Jan–Jul 2020), log-scaled so that small and large outbreaks are both visible. Output: [result/covid_choropleth.html](result/covid_choropleth.html).

### Supporting visualizations:
- **Bubble map** ([src/covid_bubble_map.py](src/covid_bubble_map.py)) — animated scatter-geo sized by √(confirmed), colored by WHO region.
- **Mortality rate** ([src/covid_mortality_rate.py](src/covid_mortality_rate.py)) — animated choropleth of deaths / confirmed, capped at 15% to prevent small-sample outliers from washing out the scale.
- **Recovery rate** ([src/covid_recovery_rate.py](src/covid_recovery_rate.py)) — animated choropleth of recovered / confirmed.
- **WHO region comparison** ([src/covid_who_regions.py](src/covid_who_regions.py)) — line chart of total confirmed cases split by WHO region over time.

### Worldometer snapshot: [src/covid_worldometer.py](src/covid_worldometer.py)
Static choropleth of cases per million population (log-scaled) plus a horizontal bar chart of the top 30 countries by critical-case rate, colored by continent. Outputs: [result/covid_worldometer.html](result/covid_worldometer.html), [result/covid_healthcare_burden.html](result/covid_healthcare_burden.html).

### USA county drill-down: [src/covid_usa_counties.py](src/covid_usa_counties.py)
Animated county-level choropleth for the United States using the plotly FIPS GeoJSON (cached locally as [us_counties_geojson.json](us_counties_geojson.json)). ~1,978 counties × weekly frames.

### Forecasting: [src/covid_forecast.py](src/covid_forecast.py)
Holt-Winters exponential smoothing fit to global daily confirmed cases, extrapolated 30 days past the dataset end with a 95% confidence band. Requires `statsmodels`.

### Interactive dashboard: [src/covid_dashboard.py](src/covid_dashboard.py)
A Dash app combining a global choropleth (date slider + metric radio) with a multi-country time-series line chart. Run it with `python src/covid_dashboard.py` and open http://127.0.0.1:8050.

## Data

Six CSVs in [data/](data/), from the Kaggle "Corona Virus Report" dataset:

| File | Rows | Description |
|---|---|---|
| `covid_19_clean_complete.csv` | 49,068 | Daily province/state-level records, Jan 22 – Jul 27 2020 |
| `full_grouped.csv` | 35,156 | Country-level daily aggregates with `New cases` |
| `day_wise.csv` | 188 | Global daily totals (used by the forecast) |
| `country_wise_latest.csv` | 187 | End-of-period country snapshot |
| `worldometer_data.csv` | 209 | Population / healthcare-capacity snapshot |
| `usa_county_wise.csv` | 627,920 | Daily US county records with FIPS codes |

## Installation

```bash
pip install pandas numpy plotly dash statsmodels
```

Python 3.8+ recommended. `statsmodels` is only needed for the forecast script; `dash` only for the dashboard.

## Running

Run any visualization script directly. Each one reads from [data/](data/), writes an HTML into [result/](result/), and auto-opens it in the browser:

```bash
python src/covid.py
python src/covid_bubble_map.py
python src/covid_mortality_rate.py
# ...etc.
```

For the dashboard:

```bash
python src/covid_dashboard.py
```

## Project structure

```
Covid-Test/
  src/                      # one script per visualization
  data/                     # source CSVs
  result/                   # generated HTML outputs
  shared_utils.py           # load_and_aggregate, downsample_weekly,
                            # COUNTRY_NAME_MAP, WORLDOMETER_TO_CLEAN_MAP,
                            # PLOTLY_GEO_LAYOUT
  us_counties_geojson.json  # cached FIPS GeoJSON for county maps
  covid-test-plan.md        # full implementation & expansion plan
```

## Notes

- **Country name mapping.** Several names in the source data differ from Plotly's built-in `locationmode='country names'` list (`US` → `United States of America`, `Burma` → `Myanmar`, etc.). The mapping lives in `shared_utils.COUNTRY_NAME_MAP`. Worldometer uses yet a third convention, handled by `WORLDOMETER_TO_CLEAN_MAP` applied before the plotly mapping.
- **Log scaling.** Confirmed-case counts span six orders of magnitude, so choropleth color scales use `log10(clip(value, 1))` with a custom colorbar that reads back as `1 / 10 / 100 / 1K / 10K / 100K / 1M`.
- **Weekly downsampling.** Daily frames (188) are sluggish to animate; most scripts downsample to every 7th date via `downsample_weekly` for ~27 frames.
