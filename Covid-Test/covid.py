import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv('covid_19_clean_complete.csv', parse_dates=['Date'])

# aggregate province-level rows into single contry rows per date
df = df.groupby(['Country/Region', 'Date'], as_index=False).agg({
    'Confirmed': 'sum',
    'Deaths': 'sum',
    'Recovered': 'sum',
    'Active': 'sum',
    'Lat': 'mean',
    'Long': 'mean',
    'WHO Region': 'first',
})

# country name mapping

COUNTRY_NAME_MAP = {
    'Burma': 'Myanmar',
    'Cabo Verde': 'Cape Verde',
    'Congo (Brazzaville)': 'Republic of the Congo',
    'Congo (Kinshasa)': 'Democratic Republic of the Congo',
    'Cote d\'Ivoire': 'Ivory Coast',
    'Czechia': 'Czech Republic',
    'Eswatini': 'Eswatini',          # plotly 6.x accepts this; verify
    'Holy See': 'Vatican City',       # too small to render but map anyway
    'Korea, South': 'South Korea',    # already 'South Korea' in data — defensive
    'North Macedonia': 'North Macedonia',  # plotly 6.x accepts; verify
    'Taiwan*': 'Taiwan',
    'US': 'United States of America',
    'West Bank and Gaza': 'Palestine',
}
df['Country/Region'] = df['Country/Region'].replace(COUNTRY_NAME_MAP)

''' for simulation frames '''
# get sorted list of unique dates
dates = sorted(df['Date'].unique())

# use every 7th date
weekly_dates = dates[::7]

# always include the final date if not already selected
if dates[-1] not in weekly_dates:
    weekly_dates.append(dates[-1])

# filter dataframe
df = df[df['Date'].isin(weekly_dates)]

# add string column for animation frame labels
df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')

'''Log scale column for color expression in frames'''
# clip confirmed vals to min of 1 to avoid log10(0)
df['Confirmed_log'] = np.log10(df['Confirmed'].clip(lower=1))

'''
Mapped values:
- 0 cases → `log10(1)` = 0.0 (lightest color)
   - 10 cases → 1.0
   - 100 cases → 2.0
   - 1,000 cases → 3.0
   - 10,000 cases → 4.0
   - 100,000 cases → 5.0
   - 1,000,000 cases → 6.0
   - 4,290,259 cases → 6.63 (darkest color)
'''

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

'''Layout customization'''
geo=dict(
    showframe=False,          # no border around the map
    showcoastlines=True,      # keep coastlines for geographic reference
    projection_type='natural earth',  # familiar flat map projection
)
fig.update(geo=geo)

# override tick positions & labels so its human readable case counts
coloraxis_colorbar=dict(
    title='Confirmed Cases',
    tickvals=[0, 1, 2, 3, 4, 5, 6],
    ticktext=['1', '10', '100', '1K', '10K', '100K', '1M'],
)
fig.update_layout(coloraxis_colorbar=coloraxis_colorbar)

# label sliders
sliders=[dict(currentvalue=dict(prefix='Date: '))]
fig.update_layout(sliders=sliders)

fig.update_layout(height=600, width=1000)

fig.show()