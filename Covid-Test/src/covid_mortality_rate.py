import os
import sys
import numpy as np
import plotly.express as px

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from shared_utils import (
    COUNTRY_NAME_MAP,
    PLOTLY_GEO_LAYOUT,
    load_and_aggregate,
    downsample_weekly,
)

df = load_and_aggregate()
df['Country/Region'] = df['Country/Region'].replace(COUNTRY_NAME_MAP)
df = downsample_weekly(df)

# mortality rate with zero-division guard
df['Mortality_Rate'] = np.where(
    df['Confirmed'] > 0,
    df['Deaths'] / df['Confirmed'] * 100,
    0.0,
)

fig = px.choropleth(
    df,
    locations='Country/Region',
    locationmode='country names',
    color='Mortality_Rate',
    animation_frame='Date_str',
    color_continuous_scale='Reds',
    range_color=[0, 15],
    hover_name='Country/Region',
    hover_data={
        'Deaths': ':,',
        'Confirmed': ':,',
        'Mortality_Rate': ':.2f',
    },
    title='COVID-19 Mortality Rate by Country (Jan–Jul 2020)',
    labels={'Mortality_Rate': 'Mortality Rate (%)'},
)

fig.update_layout(
    geo=PLOTLY_GEO_LAYOUT,
    coloraxis_colorbar=dict(title='Mortality Rate (%)'),
    sliders=[dict(currentvalue=dict(prefix='Date: '))],
    height=600,
    width=1000,
)

fig.write_html(os.path.join(_ROOT, 'result', 'covid_mortality_rate.html'), auto_open=True)
