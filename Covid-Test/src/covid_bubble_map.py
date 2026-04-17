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

# sqrt scaling is better than log for bubble area perception
df['Confirmed_sqrt'] = np.sqrt(df['Confirmed'].clip(lower=0))

fig = px.scatter_geo(
    df,
    lat='Lat',
    lon='Long',
    size='Confirmed_sqrt',
    color='WHO Region',
    animation_frame='Date_str',
    size_max=40,
    projection='natural earth',
    hover_name='Country/Region',
    hover_data={
        'Confirmed': ':,',
        'Deaths': ':,',
        'Confirmed_sqrt': False,
        'Lat': False,
        'Long': False,
    },
    title='COVID-19 Confirmed Cases — Bubble Map (Jan–Jul 2020)',
)

fig.update_layout(
    geo=PLOTLY_GEO_LAYOUT,
    sliders=[dict(currentvalue=dict(prefix='Date: '))],
    height=600,
    width=1000,
)

fig.write_html(os.path.join(_ROOT, 'result', 'covid_bubble_map.html'), auto_open=True)
