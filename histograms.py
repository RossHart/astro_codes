from sklearn.neighbors.kde import KernelDensity
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def cross_validate(test_data,bandwidths,n_folds=5):
    params = {'bandwidth': bandwidths}
    kf = KFold(n=len(test_data),n_folds=n_folds,shuffle=True,random_state=0)
    grid = GridSearchCV(KernelDensity(), params,cv=kf)
    grid.fit(test_data)
    return grid.best_estimator_.bandwidth,grid


def kde_histogram(x,x_range=None,bandwidth=None,fill=False,fill_properties=None,
                  line_properties=None,n_folds=3,printout=False,N_max=1000,zorder=0):
    '''
    --- A 1D method for plotting a kernel density estimate (rather than a histogram,
    for example) ---
    
    Inputs:
    -------
    x: x data
    
    x_range: range of the data. If None, then all of the *finite* data is used.
    
    fill: if True, the histogram will have a fill colour.
    
    fill_properties: _dictionary_ of terms for the histogram fill. Can take the keys
    'color' and 'alpha'. Default is 'k' and 0.5.
    
    line_properties: _dictionary_ of terms for the line properties. Can have the 
    keys 'color', 'alpha', 'linewidth', and 'linestyle' (defaults: 'k', 1, 1, 'solid'). 
    
    n_folds: number of folds for the cross validation if no bandwidth is provided.
    
    printout: if True, then the optimised bandwidth will be returned.
    
    N_max: maximum number of points to do the cross-validation on. If more data points
    are provided, a random selection will be used.
    
    zorder: where to 'overlay' the plot.
    
    Outputs:
    --------
    x_range: range of the data.
    
    bandwidth: bandwidth of the KDE.
    '''
    # set the line + fill properties here:
    ####################################
    fp = {'color':'k',
          'alpha':0.5}  

    lp = {'color':'k',
          'alpha':1,
          'linewidth':1,
          'linestyle':'solid'}
    
    if line_properties != None:
        for l in line_properties.keys():
            lp[l] = line_properties[l]
    if fill_properties != None:
        for f in fill_properties.keys():
            fp[f] = fill_properties[f]
    ####################################
    
    np.random.seed(0)
    
    # keep only the finite, 'good' data, or the data that is 
    # within the range of x specified:
    if x_range == None:
        select_x = np.isfinite(x)
        x_range = [np.min(x),np.max(x)]    
    else:
        select_x = (x >= x_range[0]) & (x < x_range[1])
    x = x[select_x][:,np.newaxis]
    x_std = np.std(x) # for scaling the cross-validation inputs
    if len(x) > N_max:
        x_test = np.random.choice(x.squeeze(),size=N_max,replace=False)
        x_test = x_test[:,np.newaxis]
    else:
        x_test = x.copy()
        
    if bandwidth == None:
        N_steps = 100
        bandwidths = np.logspace(-2,0,N_steps)*x_std
        bandwidth, grid = cross_validate(x_test,bandwidths,n_folds)
        if printout:
            print('Optimal bandwidth found: {0:.3f}'.format(bandwidth))
    
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x)
    plot_x = np.linspace(x_range[0]-x_std,x_range[1]+x_std,100)[:,np.newaxis]
    plot_y = np.exp(kde.score_samples(plot_x))
    plot_x,plot_y = [plot_x.squeeze(),plot_y.squeeze()]
    
    if fill == True:
        _ = plt.fill_between(plot_x,0,plot_y,color=fp['color'],alpha=fp['alpha'],zorder=zorder)
    _ = plt.plot(plot_x,plot_y,color=lp['color'],alpha=lp['alpha']
                 ,lw=lp['linewidth'],linestyle=lp['linestyle'],zorder=zorder)
        
    return x_range,bandwidth
  
  
def kde_contour(x,y,xy_range=None,bandwidth=None,fill=False,fill_properties=None,
                line_properties=None,levels=[0.2,0.4,0.6,0.8],n_folds=3,
                printout=False,N_max=1000,zorder=0):
    '''
    ---Create a contour plot, given a set of values. A KDE is applied, with
    either a given bandwidth or one from CV method.---
    
    Inputs:
    -------
    x,y: the x and y data (1D arrays).
    
    xy_range: list or tuple of 4 values (x_min,x_max,y_min,y_max). Default is
    None, in which case the range is simply the min/max of the datasets.
    
    bandwidth: bandwidth of the KDE. If set as None, then CV method is applied
    to find the best value.
    
    fill: if True, a filled contour is created.
    
    fill_properties: _dictionary_ of terms for the contour fill. Has the values
    'colormap' and 'alpha' (default are 'Greys' and 0.5).
    
    line_properties: _dictionary_ of line properties. Has the values 'color',
    'alpha','linewidth' and 'linestyle' (defaults are 'k', 1, 1 + 'solid').
    
    levels: list of levels to plot containing a fraction of the points 
    (default is [0.2,0.4,0.6,0.8]).
    
    n_folds: number of folds in the CV data.
    
    printout: if True, the 'best' bandwidth is printed.
    
    N_max: maximum number of points to do the cross-validation on. If more data points
    are provided, a random selection will be used.
    
    zorder: where to 'overlay' the plot.    
    
    Outputs:
    --------
    
    x_grid, y_grid: x and y points for the contour.
    
    H: contour 'heights'
    
    V: levels to plot in the contour (corresponding to the 'levels' input)
    '''
    
    # set the line + fill properties here:
    ####################################
    fp = {'colormap':'Greys',
          'alpha':0.5}  

    lp = {'color':'k',
          'alpha':1,
          'linewidth':1,
          'linestyle':'solid'}
    
    if line_properties != None:
        for l in line_properties.keys():
            lp[l] = line_properties[l]
    if fill_properties != None:
        for f in fill_properties.keys():
            fp[f] = fill_properties[f]
    ####################################
    
    xy = np.array([x,y]).T
    
    # keep the data within the specified range 
    # + remove any data that isn't finite:
    if xy_range == None:
        select_xy = (np.isfinite(x)) & (np.isfinite(y))
        x_range = [np.min(x),np.max(x)]  
        y_range = [np.min(y),np.max(y)]  
    else:
        select_xy = ((x >= xy_range[0]) & (x <= xy_range[1]) &
                     (y >= xy_range[2]) & (y <= xy_range[3]))
        x_range = [xy_range[0],xy_range[1]]
        y_range = [xy_range[2],xy_range[3]]
        
    # scale the data -> get an optimum bandwidth:
    x_mean, x_std, y_mean, y_std = [np.mean(x),np.std(x),np.mean(y),np.std(y)]
    x_scaled, y_scaled = [scale(x),scale(y)]
    xy_scaled = np.array([x_scaled,y_scaled]).T
    if bandwidth == None:
        if len(xy_scaled) > N_max:
            cv_select = np.random.choice(len(xy_scaled),N_max,replace=False)
            xy_scaled_cv = xy_scaled[cv_select]
        else:
            xy_scaled_cv = xy_scaled.copy()
        N_steps = 100
        bandwidths = np.logspace(-2,0,N_steps)
        bandwidth, _ = cross_validate(xy_scaled_cv,bandwidths,n_folds)

    H, V, x_grid_scaled, y_grid_scaled = xy_kde(xy_scaled,bandwidth,levels=levels) 
    # KDE from obtained bandwidths
    x_grid = x_mean+ (x_std*x_grid_scaled)
    y_grid = y_mean + (y_std*y_grid_scaled)
    
    if fill:
        plt.contourf(x_grid,y_grid,H,levels=np.append(V,np.max(H))
                     ,cmap=fp['colormap'],alpha=fp['alpha'],zorder=zorder)
    
    plt.contour(x_grid,y_grid,H,levels=V,linewidths=lp['linewidth'],colors=lp['color'],
                linestyles=lp['linestyle'],alpha=lp['alpha'],zorder=zorder)
        
    return x_grid, y_grid, H, V


def xy_kde(xy,bandwidth,N_grid=100,levels=[0.8,0.6,0.4,0.2]):  
    
    x_edges = np.linspace(np.min(xy[:,0]),np.max(xy[:,0]),N_grid+1)
    y_edges = np.linspace(np.min(xy[:,1]),np.max(xy[:,1]),N_grid+1)
    x_centres = np.array([x_edges[b] + (x_edges[b+1]-x_edges[b])/2 
                          for b in range(N_grid)])
    y_centres = np.array([y_edges[b] + (y_edges[b+1]-y_edges[b])/2 
                          for b in range(N_grid)])
    x_grid, y_grid = np.meshgrid(x_centres,y_centres)
    xy_grid = np.array([np.ravel(x_grid),np.ravel(y_grid)]).T
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(xy)
    H = np.exp(kde.score_samples(xy_grid).reshape(N_grid,N_grid))
    # this bit is taken from the corner_plot.py method.
    ######################################
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    #####################################
    V = np.sort(V)
    
    return H, V, x_grid, y_grid
  
  
def corner(xyz, ranges=None, fill=False, hist_fill_properties=None,
           contour_fill_properties=None,line_properties=None,fig_axarr=None,
           levels=[0.2,0.4,0.6,0.8],max_n_ticks=5,zorder=0):
    '''
    ---Make a corner plot using my contour method.---
    
    Inputs:
    -------
    xyz: input _table_. The column headings will make up the axis labels.
    
    ranges: list of N_columns ranges. If set to None, then the ranges will
    simply be the ranges of the data. 
    
    fill: if True, conour+ histograms will be filled.
    
    hist_fill_properties: histogram fill properties (see the kde_histogram 
    module for details).
    
    contour_fill_properties: contour fill properties (see kde_contour for
    details).
    
    line_properties: _dictionary_ of line properties (see kde_contour or
    kde_histogram for details).
    
    fig_axarr: fig, axarr list, as returned from this module. If None, a 
    new set of axes are set.
    
    levels: contour levels (see kde_contour for details).
    
    max_n_ticks: maximum number of ticks (for MaxNLocator).
    
    zorder: order to 'overlay' the plot. Default is 0.
    
    Outputs:
    --------
    fig, axarr: figure and subplot axes. These can be used to plot an 
    equivalent set of data.
    
    '''
    
    labels = xyz.colnames
    
    if ranges == None:
        finite_xyz = np.array([np.isfinite(xyz[c]) for c in labels]).T
        keep_finite = np.min(finite_xyz,axis=1)
        xyz = xyz[keep_finite]
        ranges = [(np.min(xyz[c]),np.max(xyz[c])) for c in labels]

    # Some magic numbers for pretty axis layout.
    K = len(xyz.colnames) # no of columns
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim

    # Create a new figure if one wasn't provided.
    if fig_axarr is None:
        fig, axarr = plt.subplots(K, K, figsize=(dim, dim))
        # format the figure
        lb = lbdim / dim
        tr = (lbdim + plotdim) / dim
        fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                            wspace=whspace, hspace=whspace)
    else:
        fig, axarr = fig_axarr

    for i, x_column in enumerate(labels):
        x = xyz[x_column].data
        ax = axarr[i,i]
        plt.sca(ax)
        # Plot the histograms.
        _ = kde_histogram(x,fill=fill,fill_properties=hist_fill_properties,
                          line_properties=line_properties,zorder=zorder)

        # Set up the axes.
        plt.xlim(ranges[i])
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))

        if i < K - 1:
            ax.xaxis.set_ticks_position("top")
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            ax.set_xticklabels([])
        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            ax.set_xlabel(labels[i])
            ax.xaxis.set_label_coords(0.5, -0.3)
  
        for j, y_column in enumerate(labels):
            y = xyz[y_column].data
            ax = axarr[i,j]
            plt.sca(ax)
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                continue

            _ = kde_contour(y,x,fill=fill,fill_properties=contour_fill_properties,
                            line_properties=line_properties,levels=levels,zorder=zorder)

            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            
            if i < K - 1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                ax.set_xlabel(labels[j])
                ax.xaxis.set_label_coords(0.5, -0.3)

            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.3, 0.5)

    return fig, axarr