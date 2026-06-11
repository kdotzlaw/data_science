import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix

from statsmodels.tsa.holtwinters import SimpleExpSmoothing


'''
OUTCOME_CLASSIFIER
INPUT: data frame (loaders.load_clinical())
OUTPUT: tuple [dict, go.figure]

PURPOSE: 
    Identify which clinical factors push a patient toward death.
    Fits a logistic regression on the 10-patient clinical records (age, incubation,
    severity, ICU, ventilation) and presents the standardized coefficients as a bar
    chart. 

    The deliverable is interpretability, not prediction. Validated by leave-one-out
    accuracy and a confusion matrix, never AUC.

ACCEPTANCE CHECK:
    Top positive bars come out `age` (+0.82) > `severity` (+0.76) >
    `icu_admission` = `mechanical_ventilation` (+0.49 each); `incubation_days` is
    the lone negative bar.
'''
def outcome_classifier(df: pd.DataFrame)->tuple[dict,go.Figure]:
    # severity numerical flags
    SEVERITY ={"Mild":0,"Moderate":1,"Severe":2,"Critical":3}

    # create data frame of necessary info from df
    X = pd.DataFrame({
        'age':df['age'],
        'incubation_days':df['incubation_days'],
        'severity':df['severity'].map(SEVERITY),
        'icu_admission':df['icu_admission'],
        'mechanical_ventilation':df['mechanical_ventilation'],
    })
    y=df['deceased']

    # validate with leave-one-out
    model = make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))

    # predict w cross validation
    yPred = cross_val_predict(model,X,y,cv=LeaveOneOut())

    # get accuracy score & make confusion matrix
    acc = accuracy_score(y,yPred)
    cm=confusion_matrix(y,yPred)

    # refit model on all rows to read coefficients
    model.fit(X,y)
    coefs = pd.Series(
        model.named_steps['logisticregression'].coef_[0],
        index=X.columns
    ).sort_values()

    # create horiz bar chart -- red = toward deceased
    fig = go.Figure(go.Bar(
        x=coefs.values,
        y=coefs.index,
        orientation='h',
        marker_color=['crimson' if c > 0 else 'steelblue' for c in coefs.values],
    ))
    fig.update_layout(
        title='Outcome Classsifier Coefficients (positive -> deceased)',
        xaxis_title='Standardized Coefficient',
        yaxis_title='',
        height=400,
    )

    return {'loo_accuracy':acc, 'confusion_matrix':cm.tolist()},fig

'''
CFR_TREND
INPUT: data frame (loaders.load_country_yearly())
OUTPUT: tuple [dict, go.figure]

PURPOSE:
    Identify whether a country's case fatality rate is rising or falling
    over time. Fits a simple linear regression of CFR against year (only DRC has
    the ≥3 distinct years needed) and plots the observed points, the fitted line,
    and a confidence band. 

ACCEPTANCE CHECK:
    - DRC's slope is negative; the band is visibly wide given only ~5 points.

'''

def cfr_trend(yearly: pd.DataFrame)->tuple[dict, go.Figure]:
    # aggregate country-year multi-syndromes
    agg = (yearly.groupby(['country','year'], as_index=False)
           .agg(cases=('confirmed_cases','sum'), deaths=('deaths','sum')))

    # recompute pooled CFR
    agg['cfr'] = agg['deaths'] / agg['cases'] * 100

    # fit cfr ~ year (>= 3 distinct years for this dataset)
    fig = go.Figure()
    slopes={}
    for country, g in agg.groupby('country'):
        if g['year'].nunique() < 3:
            continue
        g = g.sort_values('year')

        # use CI band from statsmodel
        model = sm.OLS(g['cfr'], sm.add_constant(g['year'])).fit()
        slopes[country] = round(model.params['year'],3)

        # get model prediction
        pred = model.get_prediction(sm.add_constant(g['year'])).summary_frame()

        # create fig with Scatter
        fig.add_trace(go.Scatter(
            x=g['year'],
            y=g['cfr'],
            mode='markers',
            name=f"{country} (obs)"
        ))

        fig.add_trace(go.Scatter(
            x=g['year'],
            y=pred['mean'],
            mode='lines',
            name=f"{country} (fit)" 
        ))

        fig.add_trace(go.Scatter(
            x=list(g['year'])+list(g['year'][::-1]),
            y=list(pred['mean_ci_upper'])+list(pred['mean_ci_lower'][::-1]),
            fill='toself',
            fillcolor='rgba(0,0,0,0.08)',
            line_color='rgba(0,0,0,0)',
            showlegend=False,
        ))

    fig.update_layout(
        title='Case Fatality Rate Trend',
        xaxis_title='Year',
        yaxis_title='CFR %',
        height=450
    )
    return {'slopes':slopes}, fig
    
'''
MONTHLY_FORECAST
INPUT: data frame (loaders.load_monthly_trends(), country (str="Democratic Republic of the Congo" ), horizon (int=3))
OUTPUT: tuple [dict, go.figure]

PURPOSE:
    Demonstrate near-term case forecasting* on a single country's
    monthly series. Fits simple exponential smoothing and projects 3 months ahead
    with an approximate uncertainty band. 
    
    Because the series is sparse and irregular,
    the forecast is essentially flat at the last level. The point is to show
    forecasting *methodology* (and honest uncertainty), not to make an operational
    prediction.

ACCEPTANCE CHECK:
    - the forecast line is roughly flat at the last observed level 
    - the shaded band is wide relative to the point forecast
'''

def monthly_forecast(monthly: pd.DataFrame, country: str="Democratic Republic of the Congo", horizon: int=3) -> tuple[dict, go.Figure]:
    # slice to 1 country & order by date
    s = monthly[monthly['country']==country].sort_values('date')
    y = s['confirmed_cases'].astype(float).to_numpy()

    # fit exponential smoothing & forecast
    fit = SimpleExpSmoothing(y, initialization_method='estimated').fit()
    fc = fit.forecast(horizon)

    # build approx band from residual spread
    resid_std = np.std(fit.resid, ddof=1) if len(y) > 1 else float(y.std() or 1)
    upper = fc+1.96 * resid_std
    lower = np.clip(fc-1.96 * resid_std,0,None)

    # build future month labels
    last = s['date'].iloc[-1]
    future = pd.date_range(last + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')

    obs_x = s['date'].dt.strftime('%Y-%m-%d')
    fc_x = future.strftime('%Y-%m-%d')
    fig = go.Figure()

    # add traces to fig
    fig.add_trace(go.Scatter(
        x=obs_x,
        y=y,
        mode='lines+markers',
        name='observed'
    ))
    fig.add_trace(go.Scatter(
        x=fc_x,
        y=fc,
        mode='lines+markers',
        line_dash='dash',
        name='forecast'
    ))

    # add band traces to fig
    fig.add_trace(go.Scatter(
        x=list(fc_x) + list(fc_x[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill='toself',
        fillcolor='rgba(214,39,40,0.15)',
        line_color='rgba(0,0,0,0)',
        name='~95% band'
    ))

    fig.update_layout(title=f'{country}: Monthly Cases + {horizon}-Month Forecast', height=450)

    return {'forecast':fc.tolist()},fig

'''
SEVERITY_REGRESSION
INPUT: data frame (loaders.load_outbreaks()
OUTPUT: tuple [dict, go.figure]

PURPOSE:
    Test how well outbreak size can be predicted from basic
    attributes (decade, virus species, WHO-emergency status). Fits a regularized
    Ridge regression on the 7 outbreaks to predict log-scale case counts, shown as a
    predicted-vs-actual plot against a `y = x` reference line. 
    
    It's a sanity demo of
    fit quality (reported as log-space MAE), confirming large epidemics land high and
    small flare-ups cluster low.

ACCEPTANCE CHECK:
    - points track the diagonal loosely
    -  the large epidemics (2014–2016, 2018–2020) sit at the high-cases end and the small flare-ups
  cluster low
'''
def severity_regression(outbreaks: pd.DataFrame) -> tuple[dict,go.Figure]:
    
    # log transform heavy-tailed targets (keep features minimal n=7)

    # use ridge regularization for the tiny n with leave-one-out
    return