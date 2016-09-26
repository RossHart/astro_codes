import numpy as np
import math
from astropy.table import Table, column
from scipy.stats import binned_statistic
from astroML.resample import bootstrap

def get_line(x,y,bins=12,ranges=None,use_bootstrap=False,percentiles=(16,84)):

    def percentile_function(x):
        lower = np.percentile(x,percentiles[0])
        median = np.median(x)
        upper = np.percentile(x,percentiles[1])
        return lower, median, upper
    
    if ranges == None:
        ranges = (np.min(x),np.max(x),np.min(y),np.max(y))
        
    if isinstance(bins, int) == True:
        bins = np.linspace(ranges[0],ranges[1],bins+1)
    xbin_centres = [bins[i]+(bins[i+1]-bins[i])/2 for i in range(len(bins)-1)]
    x_bins, _, bin_assignment = binned_statistic(x,x,bins=bins,statistic='median')
    
    x_medians = np.array([])
    x_uppers = np.array([])
    x_lowers = np.array([])
    y_medians = np.array([])
    y_uppers = np.array([])
    y_lowers = np.array([])
    for b in np.unique(bin_assignment):
        in_bin = bin_assignment == b
        x_bin = x[in_bin]
        y_bin = y[in_bin]
        
        if use_bootstrap == True:
            x_lower, x_median, x_upper = bootstrap(x_bin,10,percentile_function,random_state=0)
            y_lower, y_median, y_upper = bootstrap(y_bin,10,percentile_function,random_state=0)
        else:
            x_lower, x_median, x_upper = percentile_function(x_bin)
            y_lower, y_median, y_upper = percentile_function(y_bin)
        
        x_lowers = np.append(x_lowers,x_lower)
        x_medians = np.append(x_medians,x_median)
        x_uppers = np.append(x_uppers,x_upper)
        y_lowers = np.append(y_lowers,y_lower)
        y_medians = np.append(y_medians,y_median)
        y_uppers = np.append(y_uppers,y_upper)
            
    stats_table = Table()
    stats_table['x'] = x_medians
    stats_table['x_upper'] = x_uppers
    stats_table['x_lower'] = x_lowers
    stats_table['y'] = y_medians
    stats_table['y_upper'] = y_uppers
    stats_table['y_lower'] = y_lowers
    stats_table['x_centres'] = xbin_centres
    
    return stats_table, bins