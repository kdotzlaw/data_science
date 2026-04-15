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

# log scale column for color expression in frames
df['Confirmed_log'] = np.log10(df['Confirmed'].clip(lower=1))

# animated choropleth
fig = px.choropleth(
    df,
    locations='Country/Region',
    locationmode='country names',
    color='Confirmed_log',
    animation_frame='Date_str',
    color_continuous_scale='YlOrRd',
    range_color=[0, df['Confirmed_log'].max()],
    hover_name='Country/Region',
    hover_data={'Confirmed': ':,', 'Deaths': ':,', 'Recovered': ':,', 'Confirmed_log': False},
    title='COVID-19 Confirmed Cases by Country (Jan\u2013Jul 2020)',
    labels={'Confirmed_log': 'Confirmed Cases'},
)

fig.update_layout(
    geo=PLOTLY_GEO_LAYOUT,
    coloraxis_colorbar=dict(
        title='Confirmed Cases',
        tickvals=[0, 1, 2, 3, 4, 5, 6],
        ticktext=['1', '10', '100', '1K', '10K', '100K', '1M'],
    ),
    sliders=[dict(currentvalue=dict(prefix='Date: '))],
    height=600,
    width=1000,
)

fig.write_html('covid_choropleth.html', auto_open=True)
