#!/usr/bin/python3.8
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily
import sklearn.cluster
import numpy as np
import os
import sys
# muzeze pridat vlastni knihovny

def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani"""
    df = df.dropna()
    
    return df


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Vykresleni grafu s dvemi podgrafy podle lokality nehody """


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru """



if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    # crs="EPSG:5514"
    file = sys.argv[0] 
    path_project = os.path.abspath(file+"/..")
    data = pd.read_pickle(path_project + "/"+ 'accidents.pkl.gz', compression='gzip')
    

    gdf = make_geo(data)
    print(gdf.tail())
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)

