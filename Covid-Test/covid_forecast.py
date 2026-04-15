import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv('day_wise.csv', parse_dates=['Date'])

# fit Holt-Winters exponential smoothing (additive trend, no seasonality)
model = ExponentialSmoothing(
    df['Confirmed'],
    trend='add',
    seasonal=None,
    initialization_method='estimated',
)
fit = model.fit()

# forecast 30 days beyond the last date
forecast_steps = 30
forecast = fit.forecast(steps=forecast_steps)

# build forecast date range
last_date = df['Date'].iloc[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)

# confidence intervals from residual std
std_resid = fit.resid.std()
upper = forecast + 1.96 * std_resid
lower = forecast - 1.96 * std_resid

# build figure
fig = go.Figure()

# historical line
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Confirmed'],
    mode='lines',
    name='Historical',
    line=dict(color='steelblue'),
))

# forecast line
fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=forecast,
    mode='lines',
    name='Forecast',
    line=dict(color='tomato', dash='dash'),
))

# 95% CI upper bound
fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=upper,
    mode='lines',
    name='95% CI Upper',
    line=dict(width=0),
    showlegend=False,
))

# 95% CI lower bound (filled to upper)
fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=lower,
    mode='lines',
    name='95% CI',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(255,99,71,0.2)',
))

fig.update_layout(
    title='COVID-19 Global Confirmed Cases — 30-Day Forecast (Holt-Winters)',
    xaxis_title='Date',
    yaxis_title='Confirmed Cases',
    height=600,
    width=1000,
    hovermode='x unified',
)

fig.write_html('covid_forecast.html', auto_open=True)
