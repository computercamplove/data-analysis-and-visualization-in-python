#!/usr/bin/env python3.8
# coding=utf-8
#%%
from itertools import groupby
from matplotlib import pyplot as plt
from numpy.lib.shape_base import column_stack
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import numpy as np
import os
import sys
import math

def get_dataframe(filename: str = "accidents.pkl.gz", 
                            verbose: bool = False) -> pd.DataFrame:
    #find path to file.gzip to save as data frame 'data' 
    file = sys.argv[0] 
    path_project = os.path.abspath(file+"/..")
    data = pd.read_pickle(path_project + "/"+ filename, compression='gzip')
    df = data.rename(columns={'p2a': 'date'})

    #asign new data types (datetime and cycle for category data type)
    df['date']=df['date'].astype('datetime64[M]')
    for column in ['k', 't', 'p', 'q', 'h', 'p7', 'p8', 'p9', 'p10', 'p12',
                    'p15', 'p16', 'p17', 'p18', 'p21', 'p22', 'p58', 'p19', 
                    'p20', 'p23', 'p24', 'p27', 'p28', 'p35', 'p39', 'p44', 
                    'p45a', 'p47', 'p48a', 'p49', 'p50a', 'p50b', 'p51', 'p52',
                    'p55a', 'p5a', 'p6','p11','region']:
        df[column] = df[column].astype('category')
    
    #option to check old and new size of dataframe
    if verbose == True:
        n = format(df.memory_usage(deep=True).sum()/1048576, '.1f')
        o = format(data.memory_usage(deep=True).sum()/1048576, '.1f')
        print("orig_size= {} MB\nnew_size= {} MB".format(o, n))
    
    return df


def plot_conseq(df: pd.DataFrame, fig_location: str = None,
                show_figure: bool = False):

    d = df[['region', 'p13a', 'p13b', 'p13c']].copy()
    d['total'] = 1 #for order
    region_grp = d.groupby('region').sum()
    #variable for set order for every sublot
    order = region_grp.sort_values('total', ascending=False).index
    sns.set_style("darkgrid")
    #first subplot
    plt.figure(figsize=(8,10))
    plt.subplot(4,1,1)
    p1 = sns.barplot(x=region_grp.index, 
                    y=region_grp['p13a'].values, 
                    data=region_grp,palette=("Blues_d"), 
                    order=order)
    p1.set(xticklabels=[])
    p1.set(title='Úmrtí')
    p1.set(xlabel=None)
    p1.set(ylabel='Počet')
    #second subplot
    plt.subplot(4,1,2)
    p2 = sns.barplot(x=region_grp.index, 
                    y=region_grp['p13b'].values, 
                    data=region_grp, 
                    palette=("Blues_d"),
                    order=order)
    p2.set(xticklabels=[])
    p2.set(title='Těžce ranění')
    p2.set(xlabel=None)
    p2.set(ylabel='Počet')
    #third subplot
    plt.subplot(4,1,3)
    p3 = sns.barplot(x=region_grp.index, 
                    y=region_grp['p13c'].values, 
                    data=region_grp,palette=("Blues_d"),
                    order=order)
    p3.set(xticklabels=[])
    p3.set(title='Lehce ranění')
    p3.set(xlabel=None)
    p3.set(ylabel='Počet')
    #fourth subplot
    plt.subplot(4,1,4)
    p4 = sns.barplot(x=region_grp.index, 
                    y=region_grp['total'].values, 
                    data=region_grp,palette=("Blues_d"),
                    order=order)
    p4.set(title='Celkem nehod')
    p4.set(xlabel=None)
    p4.set(ylabel='Počet')
    
    plt.tight_layout(pad=1.0)
    #options to save and show plot
    if fig_location != None:
        cwd = os.getcwd()
        plt.savefig(cwd +'/'+fig_location)
    if show_figure is True:
        plt.show()


def plot_damage(df: pd.DataFrame, fig_location: str = None,
                show_figure: bool = False):

    data = df[['region', 'p53', 'p12']].copy()
    data['p12'] = data['p12'].astype('int64')
    #creating bins and labels
    bins = [99, 200, 209, 311, 414, 516, 615]
    label=['nezaviněná řidičem','nepřiměřená rychlost jízdy', 
            'nesprávné předjíždění', 'nedání přednosti v jízdě', 
            'nesprávný způsob jízdy','technická závada vozidla']
    data['p12'] = pd.cut(data['p12'], bins=bins, labels=label)
    data['p53'] = data['p53'].div(10) #tisíc Kč
    bin2 = [-1, 50, 200, 500, 1000, np.inf]
    lab2= ['< 50', '50 - 200', '200 - 500', '500 - 1000', '1000 >']
    data['p53'] = pd.cut(data['p53'], bins=bin2, labels=lab2)
    #dataframes for 4 regions
    pha = data[data['region'] == 'PHA'].sort_values('p53', ascending=True)
    hkk = data[data['region'] == 'HKK'].sort_values('p53', ascending=True)
    plk = data[data['region'] == 'PLK'].sort_values('p53', ascending=True)
    jhm = data[data['region'] == 'JHM'].sort_values('p53', ascending=True)

    sns.set_theme(style="darkgrid")
    #one figure for 4 sublots
    fig, [[axis1, axis2],[axis3, axis4]] = plt.subplots(2,2, figsize=(10,10), 
                                            sharey=True)
    sns.countplot(x=pha['p53'], hue=pha['p12'], data=pha, ax=axis1, 
                    palette="CMRmap")
    axis1.set(yscale='log') #set logarithmic y-scale 
    axis1.set(ylabel='Počet')
    axis1.set(title='PHA')
    axis1.set(xlabel=None)
    axis1.get_legend().remove()

    sns.countplot(x=hkk['p53'], hue=hkk['p12'], data=hkk, ax=axis2, 
                    palette="CMRmap")
    axis2.set(yscale='log') #set logarithmic y-scale 
    axis2.set(ylabel=None)
    axis2.set(xlabel=None)
    axis2.set(title='HKK')
    axis2.get_legend().remove()

    sns.countplot(x=plk['p53'], hue=plk['p12'], data=plk, ax=axis3, 
                    palette="CMRmap")
    axis3.set(yscale='log') #set logarithmic y-scale 
    axis3.set(ylabel='Počet')
    axis3.set(xlabel="Škoda [tisíc Kč]")
    axis3.set(title='PLK')
    axis3.get_legend().remove()

    sns.countplot(x=jhm['p53'], hue=jhm['p12'], data=jhm, ax=axis4, 
                    palette="CMRmap")
    axis4.set(yscale='log') #set logarithmic y-scale 
    axis4.set(ylabel=None)
    axis4.set(xlabel="Škoda [tisíc Kč]")
    axis4.set(title='JHM')
    axis4.get_legend().remove()

    #setting legend outside of figure
    handles, labels = axis1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', title="Příčina nehody",
                    bbox_to_anchor=(0.5,0.0),fancybox=False, 
                    shadow=False, ncol=3)
    plt.setp(axis1, ylim=axis3.get_ylim())
    plt.tight_layout()
    #extra size for legend
    fig.subplots_adjust(bottom=0.2)
     #options to save and show plot
    if fig_location != None:
        cwd = os.getcwd()
        plt.savefig(cwd +'/'+fig_location, bbox_inches="tight")
    if show_figure is True:
        plt.show()


def plot_surface(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    
    sns.set_theme(style="darkgrid")
    data = df[['region', 'date','p16']].copy()
    data['p16'] = data['p16'].astype('str')
    data.resample('M', on='date').sum()
    data['Hue']=data['p16'].replace({'0': 'jiný stav', 
                        '1': 'suchý neznečištěný', 
                        '2':'suchý znečištěný', '3':'mokrý', '4':'bláto',
                        '5':'náledí, ujetý sníh - posypané', 
                        '6':'náledí, ujetý sníh - neposypané',
                        '7':'rozlitý olej, nafta apod.', 
                        '8':'souvislá sněhová vrstva',
                        '9':'náhlá změna stavu'}).astype('category')
    #variables for formatting date ticks
    years = mdates.YearLocator()
    years_fmt = mdates.DateFormatter('%Y')
    #dataframes for 4 regions
    msk = data[data['region'] == 'MSK']
    hkk = data[data['region'] == 'HKK']
    plk = data[data['region'] == 'PLK']
    jhm = data[data['region'] == 'JHM']

    jhm_g = jhm.groupby(['date', 'Hue']).count().dropna().astype('int64')
    msk_g = msk.groupby(['date', 'Hue']).count().dropna().astype('int64')
    hkk_g = hkk.groupby(['date', 'Hue']).count().dropna().astype('int64')
    plk_g = plk.groupby(['date', 'Hue']).count().dropna().astype('int64')

    fig, [[axis1, axis2],[axis3, axis4]] = plt.subplots(2,2,figsize=(15,10), 
                                                            sharey=True)    
    sns.lineplot(x=jhm_g.index.get_level_values('date'),
                    y= jhm_g['region'].values,
                    hue = jhm_g.index.get_level_values('Hue'),ax=axis1)
    axis1.set(ylabel='Počet')
    axis1.set(title='JHM')
    axis1.set(xlabel=None)
    axis1.get_legend().remove()
    #set x-scale sampling as 'year'
    axis1.xaxis.set_major_locator(years)
    axis1.xaxis.set_major_formatter(years_fmt)

    sns.lineplot(x=msk_g.index.get_level_values('date'),
                    y= msk_g['region'].values,
                    hue = msk_g.index.get_level_values('Hue'),ax=axis2)
    axis2.set(ylabel='Počet')
    axis2.set(title='MSK')
    axis2.set(xlabel=None)
    axis2.get_legend().remove()
    #set x-scale sampling as 'year'
    axis2.xaxis.set_major_locator(years)
    axis2.xaxis.set_major_formatter(years_fmt)

    sns.lineplot(x=hkk_g.index.get_level_values('date'),
                    y= hkk_g['region'].values,
                    hue = hkk_g.index.get_level_values('Hue'),ax=axis3)
    axis3.set(ylabel='Počet')
    axis3.set(title='HKK')
    axis3.get_legend().remove()
    #set x-scale sampling as 'year'
    axis3.xaxis.set_major_locator(years)
    axis3.xaxis.set_major_formatter(years_fmt)
    axis3.set(xlabel="Datum vzniku nehody")

    sns.lineplot(x=plk_g.index.get_level_values('date'),
                    y= plk_g['region'].values,
                    hue = plk_g.index.get_level_values('Hue'),ax=axis4)
    axis4.set(ylabel='Počet')
    axis4.set(title='PLK')
    axis4.get_legend().remove()
    #set x-scale sampling as 'year'
    axis4.xaxis.set_major_locator(years)
    axis4.xaxis.set_major_formatter(years_fmt)
    axis4.set(xlabel="Datum vzniku nehody")

    #setting legend outside of figure
    handles, labels = axis1.get_legend_handles_labels()
    fig.legend(handles, labels, loc ='lower center', title = "Příčina nehody",
                    bbox_to_anchor=(0.5,0.0),fancybox = False, 
                    shadow = False, ncol = 4)
    plt.setp(axis1, ylim = axis3.get_ylim())
    plt.tight_layout()
    fig.subplots_adjust(bottom = 0.2)
     #options to save and show plot
    if fig_location != None:
        cwd = os.getcwd()
        plt.savefig(cwd +'/'+fig_location, bbox_inches = "tight")
    if show_figure is True:
        plt.show()

if __name__ == "__main__":
    pass

    df = get_dataframe("accidents.pkl.gz")
    plot_conseq(df, fig_location="01_nasledky.png", show_figure=True)
    plot_damage(df, "02_priciny.png", True)
    plot_surface(df, "03_stav.png", True)
# %%

