import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity

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


def cross_validate(test_data,bandwidths,n_folds=5):
    params = {'bandwidth': bandwidths}
    kf = KFold(n=len(test_data),n_folds=n_folds,shuffle=True,random_state=0)
    grid = GridSearchCV(KernelDensity(), params,cv=kf)
    grid.fit(test_data)
    return grid.best_estimator_.bandwidth,grid


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