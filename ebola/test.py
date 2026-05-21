from loaders import load_outbreaks
from charts import outbreak_gantt
from utils import save_figure


if __name__=='__main__':
    fig = outbreak_gantt(load_outbreaks())
    save_figure(fig,"outbreak_gantt")
