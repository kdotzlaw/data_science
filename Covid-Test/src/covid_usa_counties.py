import os
import json
import numpy as np
import pandas as pd
import plotly.express as px

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Download and cache US counties GeoJSON ---
GEOJSON_FILE = os.path.join(_ROOT, 'us_counties_geojson.json')
GEOJSON_URL = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'

if not os.path.exists(GEOJSON_FILE):
    from urllib.request import urlretrieve
    print('Downloading US counties GeoJSON (~17 MB)...')
    urlretrieve(GEOJSON_URL, GEOJSON_FILE)
    print('Done.')

with open(GEOJSON_FILE) as f:
    counties = json.load(f)

# --- Load and prepare data ---
df = pd.read_csv(os.path.join(_ROOT, 'data', 'usa_county_wise.csv'))
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')

# drop null FIPS rows (aggregate entries like "Kansas City", correctional facilities)
df = df.dropna(subset=['FIPS'])

# format FIPS as 5-digit zero-padded string
df['FIPS'] = df['FIPS'].astype(int).astype(str).str.zfill(5)

# --- Downsample to weekly ---
dates = sorted(df['Date'].unique())
weekly_dates = list(dates[::7])
if dates[-1] not in weekly_dates:
    weekly_dates.append(dates[-1])

df = df[df['Date'].isin(weekly_dates)].copy()
df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')

# --- Log scale ---
df['Confirmed_log'] = np.log10(df['Confirmed'].clip(lower=1))

# --- Build choropleth ---
fig = px.choropleth(
    df,
    geojson=counties,
    locations='FIPS',
    color='Confirmed_log',
    animation_frame='Date_str',
    color_continuous_scale='YlOrRd',
    range_color=[0, df['Confirmed_log'].max()],
    scope='usa',
    hover_name='Admin2',
    hover_data={
        'Province_State': True,
        'Confirmed': ':,',
        'Deaths': ':,',
        'FIPS': True,
        'Confirmed_log': False,
    },
    title='COVID-19 Confirmed Cases by US County (Jan–Jul 2020)',
    labels={'Confirmed_log': 'Confirmed Cases'},
)

fig.update_layout(
    geo=dict(scope='usa', showlakes=True),
    coloraxis_colorbar=dict(
        title='Confirmed Cases',
        tickvals=[0, 1, 2, 3, 4, 5],
        ticktext=['1', '10', '100', '1K', '10K', '100K'],
    ),
    sliders=[dict(currentvalue=dict(prefix='Date: '))],
    height=600,
    width=1000,
)

fig.write_html(os.path.join(_ROOT, 'result', 'covid_usa_counties.html'), auto_open=True)
