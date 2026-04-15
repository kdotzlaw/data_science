import numpy as np
import plotly.express as px
import pandas as pd
from shared_utils import (
    WORLDOMETER_TO_CLEAN_MAP,
    COUNTRY_NAME_MAP,
    PLOTLY_GEO_LAYOUT,
)

wm = pd.read_csv('worldometer_data.csv')
wm = wm.dropna(subset=['Population'])

# map worldometer names → csv names → plotly names
wm['Country/Region'] = wm['Country/Region'].replace(WORLDOMETER_TO_CLEAN_MAP)
wm['Country/Region'] = wm['Country/Region'].replace(COUNTRY_NAME_MAP)

# --- 2a: Cases Per Million (static choropleth) ---

wm['Cases_Per_Million'] = wm['TotalCases'] / wm['Population'] * 1e6
wm['CPM_log'] = np.log10(wm['Cases_Per_Million'].clip(lower=1))

fig_cpm = px.choropleth(
    wm,
    locations='Country/Region',
    locationmode='country names',
    color='CPM_log',
    color_continuous_scale='Viridis',
    hover_name='Country/Region',
    hover_data={
        'Cases_Per_Million': ':.0f',
        'TotalCases': ':,',
        'Population': ':,',
        'CPM_log': False,
    },
    title='COVID-19 Cases Per Million by Country (Worldometer Snapshot)',
    labels={'CPM_log': 'Cases Per Million'},
)

fig_cpm.update_layout(
    geo=PLOTLY_GEO_LAYOUT,
    coloraxis_colorbar=dict(
        title='Cases Per Million',
        tickvals=[0, 1, 2, 3, 4],
        ticktext=['1', '10', '100', '1K', '10K'],
    ),
    height=600,
    width=1000,
)

fig_cpm.write_html('covid_worldometer.html', auto_open=True)

# --- 2b: Healthcare Burden (horizontal bar chart) ---

burden = wm.dropna(subset=['Serious,Critical']).copy()
burden['Critical_Rate'] = burden['Serious,Critical'] / burden['TotalCases'] * 100
burden = burden.nlargest(30, 'Critical_Rate')

fig_burden = px.bar(
    burden,
    x='Critical_Rate',
    y='Country/Region',
    orientation='h',
    color='Continent',
    hover_data={
        'Serious,Critical': ':,',
        'TotalCases': ':,',
        'Critical_Rate': ':.2f',
    },
    title='COVID-19 Healthcare Burden — Top 30 Countries by Critical Case Rate',
    labels={'Critical_Rate': 'Critical Rate (%)'},
)

fig_burden.update_layout(
    yaxis={'categoryorder': 'total ascending'},
    height=800,
    width=1000,
)

fig_burden.write_html('covid_healthcare_burden.html', auto_open=True)
