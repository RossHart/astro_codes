from astropy.table import Table
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

def bin_by_column(column, nbins, fixedcount=True):
    sorted_indices = np.argsort(column)
    if fixedcount:
        bin_edges = np.linspace(0, 1, nbins + 1)
        bin_edges[-1] += 1
        values = np.empty(len(column))
        values[sorted_indices] = np.linspace(0, 1, len(column))
        bins = np.digitize(values, bins=bin_edges)
    else:
        bin_edges = np.linspace(np.min(column),np.max(column), nbins + 1)
        bin_edges[-1] += 1
        values = column
        bins = np.digitize(values, bins=bin_edges)
    x, b, n = binned_statistic(values, column, bins=bin_edges)
    return x, bins


def get_fraction_and_error(column_data,bins):
    
    bv = np.unique(bins)
    Nb = len(bv)
    values = np.zeros((Nb,2))
    
    for n,b in enumerate(bv):
        col_z = column_data[bins == b]
        values[n] = [np.mean(col_z),np.std(col_z)/np.sqrt(len(col_z))]
        
    values = Table(values,names=('mean','sigma'))
        
    return values


x = np.linspace(0,100,100)
y = x**2 + 10*x*np.random.randn(len(x))

x_plot, bins = bin_by_column(x,10,fixedcount=True)
values = get_fraction_and_error(y,bins)
_ = plt.plot(x_plot,values['mean'])
_ = plt.fill_between(x_plot,values['mean']-values['sigma'],values['mean']+values['sigma'],alpha=0.5)
_ = plt.scatter(x,y)
plt.show()