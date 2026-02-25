import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__=='__main__':
    # csv to dataframe
    df = pd.read_csv('post_covid_health_effects_dataset.csv')
    print(df.head())