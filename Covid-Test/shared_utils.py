import os
import pandas as pd

# Country name mapping: csv names → plotly-compatible names
COUNTRY_NAME_MAP = {
    'Burma': 'Myanmar',
    'Cabo Verde': 'Cape Verde',
    'Congo (Brazzaville)': 'Republic of the Congo',
    'Congo (Kinshasa)': 'Democratic Republic of the Congo',
    'Cote d\'Ivoire': 'Ivory Coast',
    'Czechia': 'Czech Republic',
    'Eswatini': 'Eswatini',
    'Holy See': 'Vatican City',
    'Korea, South': 'South Korea',
    'North Macedonia': 'North Macedonia',
    'Taiwan*': 'Taiwan',
    'US': 'United States of America',
    'West Bank and Gaza': 'Palestine',
}

# Worldometer country names → covid_19_clean_complete.csv names
# Apply this BEFORE COUNTRY_NAME_MAP
WORLDOMETER_TO_CLEAN_MAP = {
    'USA': 'US',
    'UK': 'United Kingdom',
    'UAE': 'United Arab Emirates',
    'S. Korea': 'South Korea',
    'CAR': 'Central African Republic',
    'Congo': 'Congo (Brazzaville)',
    'DRC': 'Congo (Kinshasa)',
    'Ivory Coast': "Cote d'Ivoire",
    'Myanmar': 'Burma',
    'Palestine': 'West Bank and Gaza',
    'St. Vincent Grenadines': 'Saint Vincent and the Grenadines',
    'Taiwan': 'Taiwan*',
    'Vatican City': 'Holy See',
}

# Shared geo layout for plotly maps
PLOTLY_GEO_LAYOUT = dict(
    showframe=False,
    showcoastlines=True,
    projection_type='natural earth',
)

# Resolve data directory relative to this file
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_and_aggregate():
    """Load covid_19_clean_complete.csv and aggregate province-level rows to country level."""
    csv_path = os.path.join(_DATA_DIR, 'covid_19_clean_complete.csv')
    df = pd.read_csv(csv_path, parse_dates=['Date'])

    df = df.groupby(['Country/Region', 'Date'], as_index=False).agg({
        'Confirmed': 'sum',
        'Deaths': 'sum',
        'Recovered': 'sum',
        'Active': 'sum',
        'Lat': 'mean',
        'Long': 'mean',
        'WHO Region': 'first',
    })

    return df


def downsample_weekly(df):
    """Select every 7th date plus the final date; add Date_str column."""
    dates = sorted(df['Date'].unique())
    weekly_dates = list(dates[::7])

    if dates[-1] not in weekly_dates:
        weekly_dates.append(dates[-1])

    df = df[df['Date'].isin(weekly_dates)].copy()
    df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')

    return df
