from astropy.table import Table, column
import numpy as np
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from contours import kde_contour

def make_figure(xlabel='x', ylabel='y'):
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(3,3)
    ax0 = plt.subplot(gs[0,0:2])
    ax1 = plt.subplot(gs[1:,2])
    ax2 = plt.subplot(gs[1:,0:2])
    axarr = [ax0, ax1, ax2]
    plt.subplots_adjust(hspace=0,wspace=0)
    if xlabel != None:
        axarr[2].set_xlabel(xlabel)
    if ylabel != None:
        axarr[2].set_ylabel(ylabel)
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])
    axarr[1].set_xticks([])
    axarr[1].set_yticks([])
    return fig, axarr


def histogram(data,bins,fill_properties,line_properties,
              fill=False,weights=None,normed=True,orientation='horizontal'):
    
    if fill == True:
        _ = plt.hist(data,bins,normed=normed,histtype='stepfilled',
                     weights=weights,orientation=orientation,
                     color=fill_properties['color'],
                     alpha=fill_properties['alpha'])
    
    _ = plt.hist(data,bins,normed=normed,histtype='step',
                 weights=weights,orientation=orientation,
                 color=line_properties['color'],
                 linewidth=line_properties['linewidth'],
                 linestyle=line_properties['linestyle'],
                 alpha=line_properties['alpha'])
    return None


def compare_xy_data(x,y,xy_range,N_bins,weights=None,xlabel='x',ylabel='y',
                    fill=False,contour_fill_properties=None,axarr=None,fig=None,
                    hist_fill_properties=None,line_properties=None,
                    bandwidth=None,levels=[0.2,0.4,0.6,0.8]):
  
    '''
    Inputs:
    ------
    x, y : x-data, y-data (list, array, column etc.)
    
    xy_range: tuple of 4-values (xmin,xmax,ymin,ymax).
    
    N_bins: tuple of (number of xbins, number of ybins)
    
    weights: weighting to each point, same length as x or y. Default is None,
    meaning that all points are weighted the same.
    
    xlabel,ylabel: labels for the axes
    
    fill: if True, then filled contours and histograms will be plotted.
   
    
    
    '''
    
    xmin, xmax, ymin, ymax = xy_range
    
    if xy_range == None:
        x_hist = x
        y_hist = y
        x_contour = x
        y_contour = y
        w_x, w_y, w_c = [weights, weights, weights]
    else:
        x_ok = (x >= xmin) & (x <= xmax)
        y_ok = (y >= ymin) & (y <= ymax)
        x_hist = x[x_ok]
        y_hist = y[y_ok]
        x_contour = x[(x_ok) & (y_ok)]
        y_contour = y[(x_ok) & (y_ok)]
        if weights != None:
            w_x = weights[x_ok]
            w_y = weights[y_ok]
            w_c = weights[(x_ok) & (y_ok)]
        else:
            w_x, w_y, w_c = [None,None,None]
    
    c_fp = {'colormap':'Greys',
           'alpha':0.5}  

    h_fp = {'color':'k',
            'alpha':1}
    
    lp = {'color':'k',
          'alpha':1,
          'linewidth':1,
          'linestyle':'solid'}
    
    if line_properties != None:
        for l in line_properties.keys():
            lp[l] = line_properties[l]
    if hist_fill_properties != None:
        for f in hist_fill_properties.keys():
            h_fp[f] = hist_fill_properties[f]
    if contour_fill_properties != None:
        for f in contour_fill_properties.keys():
            c_fp[f] = contour_fill_properties[f]
    
    x_bins = np.linspace(xy_range[0],xy_range[1],N_bins[0]+1)
    y_bins = np.linspace(xy_range[2],xy_range[3],N_bins[1]+1)
    
    if axarr == None:
        fig, axarr = make_figure(xlabel,ylabel)
    
    plt.sca(axarr[0])
    _ = histogram(x_hist,x_bins,h_fp,lp,fill,weights=w_x,orientation='vertical')
    plt.sca(axarr[1])
    _ = histogram(y_hist,y_bins,h_fp,lp,fill,weights=w_y)
    plt.sca(axarr[2])
    #_ = plt.scatter(x_contour,y_contour,alpha=0.01)
    _ = kde_contour(x_contour,y_contour,xy_range,bandwidth,fill,c_fp,lp,levels,weights=w_c)
    bandwidth = _[-1]
    
    axarr[0].set_xlim(xmin,xmax)
    axarr[1].set_ylim(ymin,ymax)
    axarr[2].axis((xmin,xmax,ymin,ymax))
    
    return fig, axarr, bandwidth