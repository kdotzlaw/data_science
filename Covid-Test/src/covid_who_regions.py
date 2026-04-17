import os
import pandas as pd
import plotly.express as px

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(_ROOT, 'data', 'full_grouped.csv'), parse_dates=['Date'])

# aggregate to WHO Region level per date
df = df.groupby(['Date', 'WHO Region'], as_index=False).agg({
    'Confirmed': 'sum',
    'Deaths': 'sum',
    'Recovered': 'sum',
    'New cases': 'sum',
})

fig = px.line(
    df,
    x='Date',
    y='Confirmed',
    color='WHO Region',
    hover_data={'Confirmed': ':,'},
    title='COVID-19 Total Confirmed Cases by WHO Region',
)

fig.update_layout(height=600, width=1000)

fig.write_html(os.path.join(_ROOT, 'result', 'covid_who_regions.html'), auto_open=True)
