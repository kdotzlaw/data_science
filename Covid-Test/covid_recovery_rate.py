import numpy as np
import plotly.express as px
from shared_utils import (
    COUNTRY_NAME_MAP,
    PLOTLY_GEO_LAYOUT,
    load_and_aggregate,
    downsample_weekly,
)

df = load_and_aggregate()
df['Country/Region'] = df['Country/Region'].replace(COUNTRY_NAME_MAP)
df = downsample_weekly(df)

# recovery rate with zero-division guard
df['Recovery_Rate'] = np.where(
    df['Confirmed'] > 0,
    df['Recovered'] / df['Confirmed'] * 100,
    0.0,
)

fig = px.choropleth(
    df,
    locations='Country/Region',
    locationmode='country names',
    color='Recovery_Rate',
    animation_frame='Date_str',
    color_continuous_scale='Greens',
    range_color=[0, 100],
    hover_name='Country/Region',
    hover_data={
        'Recovered': ':,',
        'Confirmed': ':,',
        'Recovery_Rate': ':.2f',
    },
    title='COVID-19 Recovery Rate by Country (Jan–Jul 2020) — Data may be incomplete',
    labels={'Recovery_Rate': 'Recovery Rate (%)'},
)

fig.update_layout(
    geo=PLOTLY_GEO_LAYOUT,
    coloraxis_colorbar=dict(title='Recovery Rate (%)'),
    sliders=[dict(currentvalue=dict(prefix='Date: '))],
    height=600,
    width=1000,
)

fig.write_html('covid_recovery_rate.html', auto_open=True)
