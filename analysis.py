#!/usr/bin/env python3.8
# coding=utf-8

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import sys
import math
# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

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
        x = df.memory_usage(deep=True).sum()/1048576
        y = data.memory_usage(deep=True).sum()/1048576
        n = format(x, '.1f')
        o = format(y, '.1f')
        print("orig_size= {}\nnew_size= {}".format(o, n))

    return data

# Ukol 2: následky nehod v jednotlivých regionech
def plot_conseq(df: pd.DataFrame, fig_location: str = None,
                show_figure: bool = False):
    pass

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
    df = get_dataframe("accidents.pkl.gz", verbose=False)
    plot_conseq(df, fig_location="01_nasledky.png", show_figure=True)
    plot_damage(df, "02_priciny.png", True)
    plot_surface(df, "03_stav.png", True)

