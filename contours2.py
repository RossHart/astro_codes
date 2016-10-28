import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold 

#import warnings
#warnings.filterwarnings("ignore")



def cross_validate(data,bandwidths,n_folds=5):
    params = {'bandwidth': bandwidths}
    kf = KFold(n_splits=n_folds,random_state=0)
    grid = GridSearchCV(KernelDensity(), params,cv=kf)
    grid.fit(data)
    return grid.best_estimator_.bandwidth, grid


def cross_validate(test_data,bandwidths,n_folds=5):
    params = {'bandwidth': bandwidths}
    kf = KFold(n=len(test_data),n_folds=n_folds,shuffle=True,random_state=0)
    grid = GridSearchCV(KernelDensity(), params,cv=kf)
    grid.fit(test_data)
    return grid.best_estimator_.bandwidth,grid


def xy_kde(xy,bandwidth,N_grid=100,levels=[0.8,0.6,0.4,0.2],weights=None):  
    
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
    if weights is not None:
        weighting_kde = NadarayaWatson('gaussian',h=bandwidth)
        weighting_kde.fit(xy,weights)
        W = weighting_kde.predict(xy_grid).reshape((N_grid,N_grid))
        H = H*W
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
    
    return H, V, x_grid, y_grid, bandwidth


def kde_contour(x,y,xy_range=None,bandwidth=None,fill=False,fill_properties=None,
                line_properties=None,levels=[0.2,0.4,0.6,0.8],n_folds=3,
                printout=False,N_max=1000,zorder=0,weights=None,weight_method='NW'):
  
    np.random.seed(0)

    # set the line + fill properties here:
    ####################################
    fp = {'colormap':'Greys',
          'alpha':1}  

    lp = {'color':'k',
          'alpha':1,
          'linewidth':1,
          'linestyle':'solid'}
    
    if line_properties is not None:
        for l in line_properties.keys():
            lp[l] = line_properties[l]
    if fill_properties is not None:
        for f in fill_properties.keys():
            fp[f] = fill_properties[f]
    ####################################
    if weights is not None and weight_method is not 'NW':
        w_int = np.round(weights,decimals=0).astype(int)
        x = np.repeat(x,w_int)
        y = np.repeat(y,w_int)
        weights = None
    
    xy = np.array([x,y]).T
    
    # keep the data within the specified range 
    # + remove any data that isn't finite:
    if xy_range is None:
        select_xy = (np.isfinite(x)) & (np.isfinite(y))
        x_range = [np.min(x),np.max(x)]  
        y_range = [np.min(y),np.max(y)]  
    else:
        select_xy = ((x >= xy_range[0]) & (x <= xy_range[1]) &
                     (y >= xy_range[2]) & (y <= xy_range[3]))
        x_range = [xy_range[0],xy_range[1]]
        y_range = [xy_range[2],xy_range[3]]
    x, y = xy[:,0][select_xy], xy[:,1][select_xy]
    if weights is not None:
        weights = weights[select_xy]
    
    # scale the data -> get an optimum bandwidth:
    x_mean, x_std, y_mean, y_std = [np.mean(x),np.std(x),np.mean(y),np.std(y)]
    x_scaled, y_scaled = scale(x) ,scale(y)
    xy_scaled = np.array([x_scaled,y_scaled]).T
    
    if bandwidth is None:
        if len(xy_scaled) > N_max:
            cv_select = np.random.choice(len(xy_scaled),N_max,replace=False)
            xy_scaled_cv = xy_scaled[cv_select]
        else:
            xy_scaled_cv = xy_scaled.copy()
        N_steps = 100
        bandwidths = np.logspace(-2,0,N_steps)
        bandwidth, _ = cross_validate(xy_scaled_cv,bandwidths,n_folds)
    
    H, V, x_grid_scaled, y_grid_scaled, bandwidth = xy_kde(xy_scaled,bandwidth,levels=levels,
                                                           weights=weights) 
    # KDE from obtained bandwidths
    x_grid = x_mean + (x_std*x_grid_scaled)
    y_grid = y_mean + (y_std*y_grid_scaled)
    
    if fill is True:
        plt.contourf(x_grid,y_grid,H,levels=np.append(V,np.max(H))
                     ,cmap=fp['colormap'],alpha=fp['alpha'],zorder=zorder)
    
    plt.contour(x_grid,y_grid,H,levels=V,linewidths=lp['linewidth'],colors=lp['color'],
                linestyles=lp['linestyle'],alpha=lp['alpha'],zorder=zorder)
        
    return x_grid, y_grid, H, V, bandwidth
    
    