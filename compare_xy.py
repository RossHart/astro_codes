from astropy.table import Table, column
import matplotlib.gridspec as gridspec
import math
import matplotlib.pyplot as plt
import numpy as np

from contours import kde_contour, plot_contour
from histograms import histogram

def make_figure(figscale=10):
    fig = plt.figure(figsize=(figscale,figscale))
    gs = gridspec.GridSpec(3,3)
    ax0 = plt.subplot(gs[0,0:2])
    ax1 = plt.subplot(gs[1:,2])
    ax2 = plt.subplot(gs[1:,0:2])
    axarr = [ax0, ax1, ax2]
    plt.subplots_adjust(hspace=0,wspace=0)
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])
    axarr[1].set_xticks([])
    axarr[1].set_yticks([])
    return fig, axarr


def comparison_plot(x,y,x_range=None,y_range=None,x_bins='knuth',y_bins='knuth',bandwidth=None,
                    weights=None,fill=False,fig=None,axarr=None,line_properties=None,
                    histogram_fill_properties=None,contour_fill_properties=None,
                    normalise=True,levels=(0.2,0.4,0.6,0.8),figscale=10,zorder=0):
  
    '''
    Inputs:
    ------
    kde_contour(x,y,x_range=None,y_range=None,bandwidth=None,fill=False,
                line_properties=None,fill_properties=None,
                levels=[0.2,0.4,0.6,0.8],n_folds=3,N_max=1000,
                zorder=0,weights=None,plot=True):
    '''
    
    if (fig == None) | (axarr == None):
        fig, axarr = make_figure(figscale)

    plt.sca(axarr[0])
    x_bins = histogram(x,x_bins,x_range,fill,line_properties,
                       histogram_fill_properties,normalise,weights,
                       orientation='vertical')
    plt.sca(axarr[1])
    y_bins = histogram(y,y_bins,y_range,fill,line_properties,
                       histogram_fill_properties,normalise,weights,
                       orientation='horizontal')
    plt.sca(axarr[2])
    x_grid, y_grid, H, V, bandwidth = kde_contour(x,y,x_range,y_range,bandwidth,fill,
                                                  line_properties,contour_fill_properties,levels)
    
    axarr[0].set_xlim(x_range)
    axarr[1].set_ylim(y_range)
    axarr[2].set_xlim(x_range)
    axarr[2].set_ylim(y_range)
    
    contour_data = (x_grid, y_grid, H, V)
    
    return fig, axarr, contour_data, x_bins, y_bins, bandwidth