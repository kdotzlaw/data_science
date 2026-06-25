"""

"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


#---HELPER FUNCTIONS---
def _midpoint(yr):
    s = str(yr)
    if '-' in s:
        a,b = s.split('-')
        return (int(a) + int(b))//2
    return int(s)

'''
----outbreak_gantt----
creates a gantt chart from the given data frame
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

'''
----crf_vs_size----
creates a CFR vs outbreak size scatter plot from the given data frame
INPUT: data frame loaders.load_outbreaks()
OUTPUT: graph object figure
'''
def cfr_vs_size(df: pd.DataFrame) -> go.Figure:

    # derive decade label
    df = df.assign(decade=(df['start_date'].dt.year // 10 * 10).astype(str)+'s')
    
    # build scatter plot
    fig = px.scatter(
        df,
        x='cases',
        y = 'fatality_rate',
        size='deaths',
        color='decade',
        hover_name='country',
        hover_data=['virus_species','description'],
        log_x=True,
        labels={'cases':'Cases (log)','fatality_rate':'CFR (%)'}, 
    )

    # log-linear fit line
    coeffs = np.polyfit(np.log10(df['cases']),df['fatality_rate'],1)
    xs = np.logspace(np.log10(df['cases'].min()),
                     np.log10(df['cases'].max()),50)
    fig.add_trace(go.Scatter(
        x=xs,
        y=np.polyval(coeffs, np.log10(xs)),
        mode='lines',
        line_dash='dot',
        name='log-linear fit',
    ))
    fig.update_layout(title='CFR vs Outbreak Size',height=450)
    return fig

'''
----cumulative_deaths----
creates annotated cumulative deaths timeline
INPUT: data frame loaders.load_country_yearly, data frame loaders.load_outbreak_timeline
OUTPUT: graph object figure of cumulative deaths (yearly)
'''
def cumulative_deaths(yearly: pd.DataFrame, timeline: pd.DataFrame) -> go.Figure:
    # aggregate yearly deaths by country
    agg = yearly.groupby(['country','year'],as_index=False)['deaths'].sum()
    agg = agg.sort_values(['country','year'])
    
    #create aggregate cumulative-death col
    agg['cumulative_deaths'] = agg.groupby('country')['deaths'].cumsum()

    # create line chart
    fig = px.line(
        agg,
        x='year',
        y='cumulative_deaths',
        color='country',
        markers=True,
    )

    # year has ranges so collapse to midpoint
    timeline = timeline.assign(plot_year=timeline['year'].apply(_midpoint))

    # add vertical line & annotation per timeline event
    for _, ev in timeline.iterrows():
        fig.add_vline(
            x=ev['plot_year'],
            line_dash='dot',
            line_color='gray',
            opacity=0.4
        )
        fig.add_annotation(
            x=ev['plot_year'],
            y=1,
            yref='paper',
            text=(ev['notes'] or "")[:40]+'...',
            textangle=-90,
            showarrow=False,
            font_size=9,
        )
    fig.update_layout(title='Cumulative Deaths by Country',height=500)
    
    return fig

'''
----bubble_map----
INPUT: data frame loaders.load_master, data frame loaders.load_country_yearly
OUTPUT: bubble map figure
'''
def bubble_map(master: pd.DataFrame, yearly:pd.DataFrame)->go.Figure:
    # 1 row per country coords table from yearly
    coords =(
        yearly[['iso3','latitude','longitude']]
            .drop_duplicates('iso3'))
    df = master.merge(coords,on='iso3',how='left')

    # plot bubble map
    fig = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        size='total_cases',
        color='average_cfr',
        hover_data=[
            'total_outbreaks',
            'total_deaths',
            'latest_outbreak_year',
            'most_common_species'
        ],
        color_continuous_scale='Reds',
        size_max=40,
        projection='natural earth'
    )
    fig.update_geos(scope='africa')
    fig.update_layout(title='Total Cases & Average CFR by Country',height=550)
    return fig

'''
----symptom_cooccurance----
INPUT: data frame loaders.load_clinical->symptom_list
OUTPUT: heatmap
'''
def symptom_cooccurance(df:pd.DataFrame)->go.Figure:
    # built patient x symptom binary matrix
    exploded = df.explode('symptom_list')
    binary = pd.crosstab(exploded['patient_id'],exploded['symptom_list'])
    binary = (binary>0).astype(int)

    # calc co-occurance
    # diag holds symptom frequency
    cooc = binary.T @ binary

    # 0 diagonal so heatmap isnt dominated
    np.fill_diagonal(cooc.values, 0)

    # order symptoms by total frequency desc (strongest pairs top left)
    order = cooc.sum().sort_values(ascending=False).index
    cooc=cooc.loc[order,order]

    # plot heatmap
    fig = px.imshow(
        cooc,
        text_auto=True,
        aspect='auto',
        color_continuous_scale='Blues',
        labels=dict(color='Co-occurance'),
    )
    fig.update_layout(title='Symptom Co-Occurance (n=10 patients)',height=550)
    return fig


'''
----transmission_lollipop----
INPUT: data frame loaders.load_transmission_factors()->impact_rank
OUTPUT: 
'''
def transmission_lollipop(df:pd.DataFrame)->go.Figure:
    # sort by category, scores desc
    df = df.sort_values(['factor_category','evidence_score'],
                        ascending=[True,False]).reset_index(drop=True)
    # build line and scatter plot to simulate lollipop
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_shape(
            type='line',
            x0=0,
            x1=row['evidence_score'],
            y0=row['factor'],
            y1=row['factor'],
            line=dict(color='lightgray',width=2),
        )
    fig.add_trace(go.Scatter(
        x=df['evidence_score'],
        y=df['factor'],
        marker=dict(
            size=14,
            color=df['impact_rank'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(
                title='Impact',
                tickvals=[1,2,3,4],
                ticktext=['Low','Med','High','Very High']
            ),
        ),
        text=df['factor_category'],
        hovertemplate="%{y}<br>Score: %{x}<br>Category: %{text}<extra></extra>",
    ))

    # make y axis match sorted order
    fig.update_yaxes(categoryorder='array',categoryarray=df['factor'].to_list())

    fig.update_layout(
        xaxis_title='Evidence Score (1-10)',
        yaxis_title="",
        title='Transmission Risk Factors',
        height=500,
        margin=dict(l=220)
    )
    return fig