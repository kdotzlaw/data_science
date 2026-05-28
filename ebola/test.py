from loaders import load_outbreaks, load_country_yearly, load_outbreak_timeline, load_master,load_clinical,load_transmission_factors
from charts import outbreak_gantt, cfr_vs_size, cumulative_deaths, bubble_map,symptom_coocurrance,transmission_lollipop
from utils import save_figure


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
    fig = symptom_coocurrance(load_clinical())
    save_figure(fig,'symptom_cooccurance')
    '''
    fig = transmission_lollipop(load_transmission_factors())
    save_figure(fig,'transmission_factors')