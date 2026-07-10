"""
Refresh all figs in /results w/out launching Streamlit
"""
from loaders import (load_outbreaks, load_country_yearly,
                       load_outbreak_timeline, load_master,
                       load_clinical, load_transmission_factors,
                       load_monthly_trends)
from charts import (outbreak_gantt, cfr_vs_size, cumulative_deaths,
                      bubble_map, symptom_cooccurance, transmission_lollipop)
from models import (outcome_classifier, cfr_trend, monthly_forecast,
                      severity_regression)
from utils import save_figure

# save eda figs
outbreaks = load_outbreaks()
save_figure(outbreak_gantt(outbreaks),'outbreak_gantt')
save_figure(cfr_vs_size(outbreaks),'cfr_vs_size')

yearly = load_country_yearly()
timeline = load_outbreak_timeline()
save_figure(cumulative_deaths(yearly,timeline),'cumulative_deaths')

master = load_master()
save_figure(bubble_map(master,yearly),'bubble_map')

save_figure(symptom_cooccurance(load_clinical()),'symptom_cooccurrence')
save_figure(transmission_lollipop(load_transmission_factors()),'transmission_lollipop')

# phase 2 models
save_figure(outcome_classifier(load_clinical())[1],'outcome_classifier_coefs')
save_figure(cfr_trend(yearly)[1],'cfr_trend')
save_figure(monthly_forecast(load_monthly_trends())[1],'monthly_forecast')
save_figure(severity_regression(outbreaks)[1],'severity_regression')
