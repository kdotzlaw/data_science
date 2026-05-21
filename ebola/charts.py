"""

"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


'''
----outbreak_gantt----
INPUT: data frame loaders.load_outbreaks()
OUTPUT: graph object figure
'''
def outbreak_gantt(df: pd.DataFrame) -> go.Figure:
    # sort df by start_date asc
    df.sort_values("start_date")

    # build base timeline
    fig = px.timeline(
        df,
        x_start="start_date",
        x_end="end_date",
        y="country",
        color="virus_species",
        hover_data=["cases","deaths","fatality_rate","description"],

    )
    # earliest at top
    fig.update_yaxes(autorange="reversed")

    # kaleido (1.x) cannot serialize pandas Timestamp scalars to JSON, so
    # convert dates touching add_trace/add_annotation to ISO date strings.
    emerg = df[df["who_emergency"]]
    fig.add_trace(go.Scatter(
        x=emerg['start_date'].dt.strftime("%Y-%m-%d"),
        y=emerg['country'],
        mode='markers',
        marker_symbol="diamond",
        marker_color="red",
        marker_size=12,
        name="WHO emergency",
    ))

    # annotate bars with case count
    for _, row in df.iterrows():
        fig.add_annotation(
            x=row['end_date'].strftime("%Y-%m-%d"),
            y=row['country'],
            text=f"{row['cases']:,}",
            showarrow=False,
            xanchor="left",
            xshift=4,
        )
    fig.update_layout(
        height=400,
        xaxis_title="",
        yaxis_title="",
        title="Recorded Ebola Outbreaks",
    )
    return fig