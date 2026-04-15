import numpy as np
import plotly.express as px
from dash import Dash, html, dcc, Input, Output
from shared_utils import (
    COUNTRY_NAME_MAP,
    PLOTLY_GEO_LAYOUT,
    load_and_aggregate,
)

# load data once at module level (daily, not weekly — slider controls date)
df = load_and_aggregate()
df['Country/Region'] = df['Country/Region'].replace(COUNTRY_NAME_MAP)

dates = sorted(df['Date'].unique())
countries = sorted(df['Country/Region'].unique())

# slider marks: every 14th date to avoid overcrowding
slider_marks = {
    i: d.strftime('%b %d') for i, d in
    [(i, dates[i]) for i in range(len(dates)) if i % 14 == 0]
}

app = Dash(__name__)

app.layout = html.Div([
    html.H1('COVID-19 Dashboard', style={'textAlign': 'center'}),

    # --- Choropleth section ---
    html.H3('Global Map'),
    dcc.RadioItems(
        id='metric-radio',
        options=[
            {'label': 'Confirmed', 'value': 'Confirmed'},
            {'label': 'Deaths', 'value': 'Deaths'},
            {'label': 'Recovered', 'value': 'Recovered'},
            {'label': 'Active', 'value': 'Active'},
        ],
        value='Confirmed',
        inline=True,
        style={'marginBottom': '10px'},
    ),
    dcc.Graph(id='choropleth'),
    dcc.Slider(
        id='date-slider',
        min=0,
        max=len(dates) - 1,
        value=len(dates) - 1,
        marks=slider_marks,
        step=1,
    ),

    html.Hr(),

    # --- Time-series section ---
    html.H3('Time Series by Country'),
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': c, 'value': c} for c in countries],
        value=['United States of America'],
        multi=True,
        style={'marginBottom': '10px'},
    ),
    dcc.Graph(id='timeseries'),
], style={'maxWidth': '1100px', 'margin': 'auto', 'padding': '20px'})


@app.callback(
    Output('choropleth', 'figure'),
    Input('date-slider', 'value'),
    Input('metric-radio', 'value'),
)
def update_choropleth(date_idx, metric):
    selected_date = dates[date_idx]
    dff = df[df['Date'] == selected_date].copy()

    # log scale for color
    dff['color_val'] = np.log10(dff[metric].clip(lower=1))

    fig = px.choropleth(
        dff,
        locations='Country/Region',
        locationmode='country names',
        color='color_val',
        color_continuous_scale='YlOrRd',
        range_color=[0, np.log10(df[metric].clip(lower=1).max())],
        hover_name='Country/Region',
        hover_data={metric: ':,', 'color_val': False},
        labels={'color_val': metric},
    )

    fig.update_layout(
        geo=PLOTLY_GEO_LAYOUT,
        coloraxis_colorbar=dict(
            title=metric,
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            ticktext=['1', '10', '100', '1K', '10K', '100K', '1M'],
        ),
        title=f'{metric} Cases — {selected_date.strftime("%Y-%m-%d")}',
        height=500,
        margin=dict(t=50, b=0),
    )
    return fig


@app.callback(
    Output('timeseries', 'figure'),
    Input('country-dropdown', 'value'),
    Input('metric-radio', 'value'),
)
def update_timeseries(selected_countries, metric):
    if not selected_countries:
        selected_countries = ['United States of America']

    dff = df[df['Country/Region'].isin(selected_countries)]

    fig = px.line(
        dff,
        x='Date',
        y=metric,
        color='Country/Region',
        hover_data={metric: ':,'},
        title=f'{metric} Over Time',
    )

    fig.update_layout(height=400, margin=dict(t=50, b=0))
    return fig


if __name__ == '__main__':
    app.run(debug=True)
