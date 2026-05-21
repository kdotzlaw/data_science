from loaders import load_outbreaks, load_country_yearly, load_outbreak_timeline
from charts import outbreak_gantt, cfr_vs_size, cumulative_deaths
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

    fig = cumulative_deaths(load_country_yearly(),load_outbreak_timeline())
    save_figure(fig, "cumulative_deaths")