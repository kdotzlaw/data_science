from loaders import load_outbreaks, load_country_yearly, load_outbreak_timeline, load_master,load_clinical,load_transmission_factors, load_monthly_trends
from charts import outbreak_gantt, cfr_vs_size, cumulative_deaths, bubble_map,symptom_cooccurance,transmission_lollipop
from utils import save_figure
from models import outcome_classifier, cfr_trend, monthly_forecast, severity_regression

if __name__=='__main__':

    #--- test charts.py ---
    ''' test gantt chart
    fig = outbreak_gantt(load_outbreaks())
    save_figure(fig,"outbreak_gantt")
    '''

    '''test cfr vs outbreak size scatter
    fig = cfr_vs_size(load_outbreaks())
    save_figure(fig,"cfr_vs_size")
    '''

    '''test cumulative deaths fig
    fig = cumulative_deaths(load_country_yearly(),load_outbreak_timeline())
    save_figure(fig, "cumulative_deaths")
    '''
    '''test bubble map
    fig = bubble_map(load_master(), load_country_yearly())
    save_figure(fig,"bubble_map")
    '''

    '''test symptom coocurrance
    fig = symptom_cooccurance(load_clinical())
    save_figure(fig,'symptom_cooccurance')
    '''

    '''
    fig = transmission_lollipop(load_transmission_factors())
    save_figure(fig,'transmission_factors')
    '''
    #----test models.py----
    '''res = outcome_classifier(load_clinical())
    print(res[0])
    save_figure(res[1],'outcome_classifier')
    '''
    
    '''res = cfr_trend(load_country_yearly())
    print(res[0])
    save_figure(res[1],'cfr_trend')
    '''

    '''res = monthly_forecast(load_monthly_trends(), "Democratic Republic of the Congo",3)
    print(res[0])
    save_figure(res[1],'monthly_forecast_DRC')
    '''

    res = severity_regression(load_outbreaks())
    print(res[0])
    save_figure(res[1],'severity_regression')