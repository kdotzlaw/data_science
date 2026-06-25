import streamlit as st
from loaders import *
from charts import *
from utils import save_figure
import plotly.graph_objects as go
from models import *

# dashboard init
st.set_page_config(page_title='Ebola Data Explorer',layout='wide')
tab1,tab2,tab3 = st.tabs(['Explore','Methodology Demo','About'])

# load cache 1x per csv
load = st.cache_data(show_spinner=False)
outbreaks   = load(load_outbreaks)()
yearly      = load(load_country_yearly)()
timeline    = load(load_outbreak_timeline)()
master      = load(load_master)()
clinical    = load(load_clinical)()
transmission = load(load_transmission_factors)()
monthly     = load(load_monthly_trends)()
species     = load(load_virus_species)()
facts       = load(load_virus_facts)()
data_dict   = load(load_data_dictionary)()


# wrap save_fig so it only runs 1x per data change (ie cache it)
@st.cache_data(show_spinner=False)
def chart(name: str, _builder, *args)->go.Figure:
    fig = _builder(*args)
    save_figure(fig,name)
    return fig

@st.cache_data(show_spinner=False)
def model(name: str, _builder, *args)-> tuple[dict,go.Figure]:
    metrics, fig = _builder(*args)
    save_figure(fig,name)
    return metrics, fig

'''
EXPLORE TAB
- charts 1 - 6 render with fig save
- add country & risk filters to cumulative_deaths & transmission_lollipop
'''
with tab1:
    st.plotly_chart(chart(
        'outbreak_gantt',
        outbreak_gantt,
        outbreaks),
        use_container_width=True
    )
    st.plotly_chart(chart(
        'cfr_vs_size',
        cfr_vs_size,
        outbreaks),
        use_container_width=True
    )
    st.plotly_chart(chart(
        'cumulative_deaths',
        cumulative_deaths,
        yearly,
        timeline),
        use_container_width=True
    )
    st.plotly_chart(chart(
        'bubble_map',
        bubble_map,
        master,
        yearly),
        use_container_width=True
    )
    st.plotly_chart(chart(
        'symptom_cooccurance',
        symptom_cooccurance,
        clinical),
        use_container_width=True
    )
    st.plotly_chart(chart(
        'transmission_lollipop',
        transmission_lollipop,
        transmission),
        use_container_width=True
    )

    # Filtered/interactive render (doesnt write to /results)
    @st.cache_data(show_spinner=False)
    def view(_builder, *args)->go.Figure:
        return _builder(*args)
    
    '''
    CUMULATIVE_DEATHS
    - add country filters
    '''
    countries = sorted(yearly['country'].unique())
    picked = st.multiselect(
        'Countries',
        countries,
        default=countries,
        key='cumdeaths_countries'
    )
    if not picked:
        st.info('Pick at least one country')
    else:
        if set(picked)==set(countries):
            fig = chart(
                'cumulative_deaths',
                cumulative_deaths,
                yearly,
                timeline
            )
        else:
            # no save
            sub = yearly[yearly['country'].isin(picked)]
            fig = view(cumulative_deaths,sub,timeline)
        st.plotly_chart(fig, use_container_width=True)

    '''
    TRANSMISSION_LOLLIPOP
    - add risk factor filters
    '''
    cats = sorted(transmission['factor_category'].unique())
    picked = st.multiselect(
        'Risk categories',
        cats,
        default=cats,
        key='lollipop_cats'
    )
    if not picked:
        st.info('Pick at least one risk factor')
    else:
        if set(picked)==set(cats):
            fig = chart(
                'transmission_lollipop',
                transmission_lollipop,
                transmission,
            )
        else:
            # no save
            fig = view(
                transmission_lollipop,
                transmission[transmission['factor_category'].isin(picked)]
            )
        st.plotly_chart(fig, use_container_width=True)
'''
METHODOLOGY TAB
- render model figs with metrics
- render per model caveat banner
'''
with tab2:
    st.warning('Demonstration only -- n is tiny (<=10 rows per model).' \
    'These are methodology demos, not operational predictions.')

    def demo(name, builder, *args, caption=""):
        metrics, fig = model(name, builder, *args)
        st.plotly_chart(fig, use_container_width=True)

        if caption:
            st.caption(caption)
        st.json(metrics)
    
    demo(
        'outcome_classifier_coefs',
        outcome_classifier,
        clinical,
        caption='Leave One Out accuracy + confusion matrix only - never AUC at n=10'
    )
    demo(
        'cfr_trend',
        cfr_trend,
        yearly,
        caption='Only Democratic Republic of Congo clears the >=3 distinct years bar.' \
        ' Band is wide on purpose.'
    )
    demo(
        'monthly_forecast',
        monthly_forecast,
        monthly,
        caption='SES on a sparse, irregular series. Extrapolates' \
        'on the last level.'
    )
    demo(
        'severity_regression',
        severity_regression,
        outbreaks,
        caption='MAE in log10 space. This is a sanity demo,' \
        'not a benchmark.'
    )
'''
ABOUT TAB
- 
'''
with tab3:
    st.subheader('About this dataset')
    st.markdown(
        "Nine small CSVs of recorded Ebola outbreaks (161 rows total). "
          "Every model in the Methodology tab is a demonstration of method on "
          "tiny data: read directions and magnitudes, not point predictions."
    )
    st.subheader('Virus Facts')
    st.dataframe(facts, use_container_width=True)
    st.subheader('Virus Species')
    st.dataframe(species, use_container_width=True)
    st.subheader('Data Dictionary')
    st.dataframe(data_dict,use_container_width=True)