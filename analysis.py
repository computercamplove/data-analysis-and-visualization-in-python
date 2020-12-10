#!/usr/bin/env python3.8
# coding=utf-8
#pylint: disable=too-many-arguments
#%%
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import sys
import math
# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

# Ukol 1: nacteni dat
def get_dataframe(filename: str = "accidents.pkl.gz", verbose: bool = False) -> pd.DataFrame:
    file = sys.argv[0]
    path_project = os.path.abspath(file+"/..")
    data = pd.read_pickle(path_project + "/"+ filename, compression='gzip')
    #print(data.info())

    #PREPROCESSING
    df = data.rename(columns={'p2a': 'date'})
    df['date'] = pd.to_datetime(df['date'])

    for column in ['k', 't', 'p', 'q', 'h', 'p7', 'p8', 'p9', 'p10', 'p12','p15', 'p16',
                    'p17', 'p18', 'p21', 'p22', 'p58', 'p19', 'p20', 'p23', 'p24',
                    'p27', 'p28', 'p35', 'p39', 'p44', 'p45a', 'p47', 'p48a', 'p49',
                    'p50a', 'p50b', 'p51', 'p52','p55a', 'p5a', 'p6','p11','region']:
        df[column] = df[column].astype('category')
    
    if verbose == True:
        n = format(df.memory_usage(deep=True).sum()/1048576, '.1f')
        o = format(data.memory_usage(deep=True).sum()/1048576, '.1f')
        print("orig_size= {} MB\nnew_size= {} MB".format(o, n))

    return df

# Ukol 2: následky nehod v jednotlivých regionech
def plot_conseq(df: pd.DataFrame, fig_location: str = None,
                show_figure: bool = False):

    d = df[['region', 'p13a', 'p13b', 'p13c']]
    d.rename(columns={'p13a': 'date'})
    region_grp = d.groupby('region').sum()

    #p13a - umreli
    #p13b - tezce zraneni
    #p13c  - lehce zraneni
    #fig, axes = plt.subplots(3, 1, figsize=(15, 5), sharey=True)
    plt.figure(figsize=(15,10))
    plt.subplot(3,1,1)
    sns.barplot(x=region_grp.index, y=region_grp['p13a'].values)
    plt.subplot(3,1,2)
    sns.barplot(x=region_grp.index, y=region_grp['p13b'].values)
    plt.subplot(3,1,3)
    sns.barplot(x=region_grp.index, y=region_grp['p13c'].values)
    plt.tight_layout(pad=3.0)
    return region_grp['p13c']


# Ukol3: příčina nehody a škoda
def plot_damage(df: pd.DataFrame, fig_location: str = None,
                show_figure: bool = False):
    pass

# Ukol 4: povrch vozovky
def plot_surface(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    pass


if __name__ == "__main__":
    pass
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni ¨
    # funkce.
    df = get_dataframe("accidents.pkl.gz")
    df2 = plot_conseq(df, fig_location="01_nasledky.png", show_figure=True)
    print(df2)
    plot_damage(df, "02_priciny.png", True)
    plot_surface(df, "03_stav.png", True)
# %%
