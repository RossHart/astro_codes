from astropy.table import Table, column
import matplotlib.pyplot as plt
import math
import numpy as np
from bin_statistics import stats_functions, assign_bins
from scipy.stats import spearmanr
from two_distribution_tests import bhattacharyya_coefficient

class x_vs_y:
    
    def __init__(self,color='k',linewidth=2,marker='o',markersize=5,
                 x_range=None,y_range=None,equal_N=False,bins=10,location='upper left'):
        self.color = color
        self.linewidth = linewidth
        self.marker = marker
        self.markersize = markersize
        self.x_range = x_range
        self.bins = bins
        self.x_range = x_range
        self.y_range = y_range
        self.equal_N = equal_N
        self.bins = bins
        self.location = location
    
    def discrete_vs_continuous(self,ax,x,y):
        y_table = stats_functions().mean_and_error(y,x)
        y_error = y_table['mean+1sigma'] - y_table['mean']
        x_plot = np.unique(x)
        y_plot = y_table['mean']
        
        _, caps, _ = ax.errorbar(x_plot,y_plot,y_error,color=self.color,
                                 fmt=self.marker,elinewidth=self.linewidth,
                                 markersize=self.markersize)
        for cap in caps:
            cap.set_color(self.color)
            cap.set_markeredgewidth(self.linewidth)
        return None
    
    def continuous_vs_continuous(self,ax,x,y):
        bin_assignments = assign_bins(x,self.x_range,self.equal_N,self.bins)
        x_table = stats_functions().mean_and_error(x,bin_assignments)
        y_table = stats_functions().mean_and_error(y,bin_assignments)
        x_plot, y_plot = x_table['mean'], y_table['mean']
        y_error = y_table['mean+1sigma'] - y_table['mean']
        _, caps, _ = ax.errorbar(x_plot,y_plot,y_error,color=self.color,
                                 fmt=self.marker,elinewidth=self.linewidth,
                                 markersize=self.markersize)
        for cap in caps:
            cap.set_color(self.color)
            cap.set_markeredgewidth(self.linewidth)
        
        return None
    
    def continuous_vs_discrete(self,x,y):
        x_table = stats_functions().mean_and_error(x,y)
        x_error = x_table['mean+1sigma'] - x_table['mean']
        y_plot = np.unique(y)
        y_plot = x_table['mean']
        ax.plot(x_plot,y_plot,color=self.color,
                fmt=self.marker,markersize=self.markersize)
        
        return None
    
    def location_to_value(self):
        if self.location is 'upper right':
            return 0.95, 0.95, 0.95, 0.85, 'right', 'top'
        elif self.location is 'lower left':
            return 0.05, 0.15, 0.05, 0.05, 'left', 'bottom'
        elif self.location is 'lower right':
            return 0.95, 0.15, 0.95, 0.05, 'right', 'bottom'
        else:
            return 0.05, 0.95, 0.05, 0.85, 'left', 'top'
        
    def show_spearmanr(self,ax,x,y):
        x1, y1, x2, y2, ha, va = x_vs_y().location_to_value()
        r, p = spearmanr(x,y)
        r_string = r'$r_s = {}$'.format(np.round(r,decimals=2))
        p_string = '$p = {}$'.format(np.round(p,decimals=2))
        ax.text(x1,y1,r_string,ha=ha,va=va,transform=ax.transAxes)
        ax.text(x2,y2,p_string,ha=ha,va=va,transform=ax.transAxes)
        return None
    
    def show_bhattacharyya(self,ax,x,y,bhattacharyya_bins):
        x1, y1, x2, y2, ha, va = x_vs_y().location_to_value()
        unique_bins = np.unique(x)
        BC = []
        D_B = []
        for i, _ in enumerate(unique_bins[:-1]):
            bin_0, bin_1 = unique_bins[i], unique_bins[i+1]
            a, b = y[x == bin_0], y[x == bin_1]
            BC_i, D_B_i = bhattacharyya_coefficient(a,b,bins=bhattacharyya_bins)
            BC.append(BC_i)
            D_B.append(D_B_i)
        BC_string = r'$\overline{{BC}} = {}$'.format(np.round(np.mean(BC),decimals=2))
        D_B_string = '$\overline{{D_B}} = {}$'.format(np.round(np.mean(D_B),decimals=2))
        ax.text(x1,y1,BC_string,ha=ha,va=va,transform=ax.transAxes)
        ax.text(x2,y2,D_B_string,ha=ha,va=va,transform=ax.transAxes)
        return None