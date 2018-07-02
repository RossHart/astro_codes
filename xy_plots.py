from astropy.table import Table, column
import imp
from bin_statistics import assign_bins, StatsFunctions, TableStats
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import spearmanr
from two_distribution_tests import bhattacharyya_coefficient

class x_vs_y:
    
    def __init__(self,x,y,weights=None,x_range=None,y_range=None):
        self.x = x
        self.y = y
        self.weights = weights
        self.x_range = x_range
        self.y_range = y_range
        
    def discrete_vs_continuous_binned_mean(self,spread=False):
        x = self.x
        y = self.y
        w = self.weights
        
        self.x_table = Table(np.unique(x)[:,np.newaxis],names=['mean'])
        if spread is True:
            self.y_table = TableStats(y,x,w).mean_and_deviation()
        else:
            self.y_table = TableStats(y,x,w).mean_and_error()
        return self
      
    def discrete_vs_continuous_binned_median(self):
        x = self.x
        y = self.y
        w = self.weights
        
        self.x_table = Table(np.unique(x)[:,np.newaxis],names=['mean'])
        self.y_table = TableStats(y,x,w).median_and_error()
        return self
    
    def continuous_vs_continuous_binned_mean(self,bin_assignments=None,bins=10,
                                             equal_N=False,spread=False):
        x = self.x
        y = self.y
        w = self.weights
        x_range = self.x_range
        y_range = self.y_range
        if bin_assignments is None:
            bin_assignments = assign_bins(x,x_range,equal_N,bins)
        self.x_table = TableStats(x,bin_assignments,w).mean_and_error()
        if spread is True:
            self.y_table = TableStats(y,bin_assignments,w).mean_and_deviation()
        else:
            self.y_table = TableStats(y,bin_assignments,w).mean_and_error()
        return self
      
    def continuous_vs_continuous_binned_median(self,bin_assignments=None,bins=10                                               ,equal_N=False,use_dev=False):
      
        x = self.x
        y = self.y
        w = self.weights
        x_range = self.x_range
        y_range = self.y_range
        if bin_assignments is None:
            bin_assignments = assign_bins(x,x_range,equal_N,bins)
        self.x_table = TableStats(x,bin_assignments,w).median_and_error()
        if use_dev is False:
            self.y_table = TableStats(y,bin_assignments,w).median_and_error()
        else:
            self.y_table = TableStats(y,bin_assignments,w).median_and_percentile()
            self.y_table['mean-1sigma'] = self.y_table['16 percentile']
            self.y_table['mean+1sigma'] = self.y_table['84 percentile']
        return self
      
    def fraction_with_feature(self,bin_assignments=None,bins=10,equal_N=False):
        
        x = self.x
        y = self.y
        w = self.weights
        x_range = self.x_range
        y_range = self.y_range
        if bin_assignments is None:
            bin_assignments = assign_bins(x,x_range,equal_N,bins)
        self.x_table = TableStats(x,bin_assignments,w).mean_and_error()
        self.y_table = TableStats(y,bin_assignments,w).fraction_with_feature()
        return self
        
        
        
    
    def line_plot(self,ax,offset=0,**kwargs):
        x_plot = np.array(self.x_table['mean'])
        y_plot = np.array(self.y_table['mean']) + offset
        _ = ax.plot(x_plot,y_plot,**kwargs)
        return None
    
    def error_plot(self,ax,style='filled',offset=0,plus=0,**kwargs):
        x_plot = np.array(self.x_table['mean'])
        y_plot = np.array(self.y_table['mean'])
        y_lower = np.array(self.y_table['mean-1sigma']) + offset + plus
        y_upper = np.array(self.y_table['mean+1sigma']) + offset - plus
        if style is 'filled':
            _ = ax.fill_between(x_plot,y_lower,y_upper,**kwargs)
        elif style is 'lined':
            _ = ax.plot(x_plot,y_lower,**kwargs)
            _ = ax.plot(x_plot,y_upper,**kwargs)
        else:
            yerrs = [y_upper-y_plot,y_plot-y_lower]
            _ = ax.errorbar(x_plot,y_plot,yerrs,**kwargs)
        return None
    
    def spearmanr(self,ax,plot=True,printout=False,location=None,
                  spacing=0.05,x_offset=0.05,y_offset=0.05,**kwargs):
        x1, y1, x2, y2, ha, va = self.location_to_value(location,spacing,
                                                        x_offset,y_offset)
        
        r, p = spearmanr(self.x,self.y)
        r_string = r'$r_s = {}$'.format(np.round(r,decimals=2))
        if 0.001 < p <= 0.1:
            p_string = r'$p = 10^{{{}}}$'.format(int(np.round(math.log10(p),decimals=0)))
        elif 0.1 < p <= 1:
            p_string = r'$p = {}$'.format(np.round(p,decimals=2))
        else:
            p_string = r'$p < 10^{-3}$'
        if plot is True:
            ax.text(x1,y1,r_string,ha=ha,va=va,transform=ax.transAxes,**kwargs)
            ax.text(x2,y2,p_string,ha=ha,va=va,transform=ax.transAxes,**kwargs)
        if printout is True:
            print(r_string, p_string)
        return r_string, p_string
        
        
    def location_to_value(self,location,spacing=0.05,x_offset=0.05,y_offset=0.05):
        if location == 'upper right':
            return (1-x_offset, 1-y_offset,1-x_offset, 1-y_offset-spacing,
                    'right', 'top')
        elif location == 'lower left':
            return (x_offset, y_offset+spacing, x_offset, y_offset,
                    'left', 'bottom')
        elif location == 'lower right':
            return (1-x_offset, y_offset+spacing, 1-x_offset, y_offset, 
                    'right', 'bottom')
        else:
            return (x_offset, 1-y_offset, x_offset, 1-y_offset-spacing, 
                    'left', 'top')
    
    def show_bhattacharyya(self,ax,bhattacharyya_bins,location='upper right',
                           printout=False,show=True,**kwargs):
        x1, y1, x2, y2, ha, va = self.location_to_value(location)
        unique_bins = np.unique(self.x)
        BC, D_B, N = [], [], []
        for i, _ in enumerate(unique_bins[:-1]):
            bin_0, bin_1 = unique_bins[i], unique_bins[i+1]
            a, b = self.y[self.x == bin_0], self.y[self.x == bin_1]
            BC_i, D_B_i = bhattacharyya_coefficient(a,b,bins=bhattacharyya_bins)
            BC.append(BC_i)
            D_B.append(D_B_i)
            N.append(min(len(a),len(b)))
        BC_string = r'$\overline{{BC}} = {}$'.format(np.round(np.average(BC,weights=N),
                                                              decimals=2))
        D_B_string = '$\overline{{D_B}} = {}$'.format(np.round(np.average(D_B,weights=N),
                                                               decimals=2)) 
        if show == True:
            ax.text(x1,y1,BC_string,ha=ha,va=va,transform=ax.transAxes,**kwargs)
            ax.text(x2,y2,D_B_string,ha=ha,va=va,transform=ax.transAxes,**kwargs)
        if printout == True:
            print(BC_string, D_B_string)
        return None
    
    def scatter(self,ax,**kwargs):
        ax.scatter(self.x,self.y,**kwargs)
        return None
