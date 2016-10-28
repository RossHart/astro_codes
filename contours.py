import math
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity

from find_nearest import find_nearest_index
from masking import select_within_range

def cross_validate(data,bandwidths,n_folds=5):
    params = {'bandwidth': bandwidths}
    kf = KFold(n_splits=n_folds,random_state=0)
    grid = GridSearchCV(KernelDensity(), params,cv=kf)
    grid.fit(data)
    return grid.best_estimator_.bandwidth, grid


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
    
    return H, V, x_grid, y_grid, bandwidth


def set_line_properties(line_properties=None):
    lp_final = {'color':'k',
                'alpha':1,
                'linewidth':1,
                'linestyle':'solid'}
    
    if line_properties is not None:
        for l in line_properties.keys():
            lp_final[l] = line_properties[l]
    
    return lp_final


def set_contour_fill_properties(fill_properties=None):
    fp_final = {'colormap':'Greys',
                'alpha':1}  

    if fill_properties is not None:
        for f in fill_properties.keys():
            fp_final[f] = fill_properties[f]
    
    return fp_final


def scale_data(data):
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_scaled = (data-data_mean)/data_std
    return data_scaled, data_mean, data_std


def unscale_data(data_scaled,data_mean,data_std):
    data_unscaled = data_scaled*data_std + data_mean
    return data_unscaled


def find_best_bandwidth(data,bandwidths,n_folds=5):
    N_bandwidths = len(bandwidths)
    best_bandwidth, _ = cross_validate(data,bandwidths,n_folds)
    i_best = find_nearest_index(bandwidths,best_bandwidth)
    if i_best == 0:
        bandwidth_lower_bound = bandwidths[0] - (bandwidths[1]-bandwidths[0])
    else:
        bandwidth_lower_bound = bandwidths[i_best-1]  
    if i_best == N_bandwidths-1:
        bandwidth_upper_bound = bandwidths[-1] + (bandwidths[-1]-bandwidths[-2])
    else:
        bandwidth_upper_bound = bandwidths[i_best+1]
    bandwidth_range = (bandwidth_lower_bound,bandwidth_upper_bound)
    return best_bandwidth, bandwidth_range


def plot_contour(x_grid,y_grid,H,V,fill=False,zorder=0,
                 line_properties=None,fill_properties=None):
    '''
    Plot a contour of data points.
    
    Inputs:
    -------
    x_grid: 1D array of x points.
    
    y_grid: 1D array of y points.
    
    H: contour heights, should have size (N(x_grid),N(y_grid))
    
    fill: if True, then the resulting plot will be a filled contour.
    Default is False.
    
    zorder: plotting z-order, default is 0.
    
    fill_properties: Dictionary of terms for the contour fill. The 
    default options are 'colormap':'Greys' and 'alpha':1
    
    line_properties: Dictionary of terms for the contour lines. The
    default options are 'color':'k', 'linewidth':1, 'alpha':1 and 
    'linestyle':'solid'.
    '''
    
    # set the line + fill properties here:
    ####################################
    fp = set_contour_fill_properties(fill_properties)
    lp = set_line_properties(line_properties)
    
    if fill is True:
        plt.contourf(x_grid,y_grid,H,levels=np.append(V,np.max(H))
                     ,cmap=fp['colormap'],alpha=fp['alpha'],zorder=zorder)
    
    plt.contour(x_grid,y_grid,H,levels=V,linewidths=lp['linewidth'],colors=lp['color'],
                linestyles=lp['linestyle'],alpha=lp['alpha'],zorder=zorder)
    
    return None


def kde_contour(x,y,x_range=None,y_range=None,bandwidth=None,fill=False,
                line_properties=None,fill_properties=None,
                levels=[0.2,0.4,0.6,0.8],n_folds=3,N_max=1000,
                zorder=0,weights=None,plot=True):
    
    '''
    A code for finding the best bandwidth for a given dataset, and 
    plotting up the resulting contour.
    
    Inputs:
    -------
    x: x dataset, an array or astropy column
    
    y: y dataset, *same length as x*
    
    bandwidth: if None, then the function will use X-validation to
    find the 'best' option.
    
    fill: if True, then the resulting plot will be a filled contour.
    Default is False.
    
    fill_properties: Dictionary of terms for the contour fill. The 
    default options are 'colormap':'Greys' and 'alpha':1
    
    line_properties: Dictionary of terms for the contour lines. The
    default options are 'color':'k', 'linewidth':1, 'alpha':1 and 
    'linestyle':'solid'.
    
    levels: fractions of points to enclose each contour. Default is
    (0.2,0.4,0.6,0.8), ie. 20, 40, 60 and 80% of the points.
    
    n_folds: number of X-validation folds for the dataset. Default=3.
    
    N_max: finding the best bandwidth takes a lot of time if there 
    are too many points: this takes a subset of N_max randomly
    chosen points instead (default=1000).
    
    zorder: plotting z-order, default is 0.
    
    weights: apply _integer_ weighting to each point. *If none-
    integer weights are supplied, they are rounded*
    
    plot: if True, the resultsa are plotted. If False, then the 
    values are returned without being plotted.
    
    Returns:
    --------
    
    x_grid: grid of 100 x points for plotting.
    
    y_grid: grid of 100 y points for plotting.
    
    H: heights to plot, 100*100 grid.
    
    V: levels to plot, corresonding to the levels input.
    
    bandwidth: optimal found bandwidth for the dataset.
    
    *Note that x_grid, y_grid, H and V can be passed to the 
    plot_contour function in that order to produce the plot at
    any point.*
    '''
    np.random.seed(0)
    # Repeat data with weights:
    if weights is not None:
        w_int = np.round(weights,decimals=0).astype(int)
        x = np.repeat(x,w_int)
        y = np.repeat(y,w_int)

    select_x, _, x_range = select_within_range(x,x_range)
    select_y, _, y_range = select_within_range(y,y_range)
    select_xy = (select_x) & (select_y)
    xy_plot = (np.array([x,y]).T)[select_xy]
    x_scaled, x_mean, x_std = scale_data(xy_plot[:,0])
    y_scaled, y_mean, y_std = scale_data(xy_plot[:,1])
    xy_scaled = np.array([x_scaled,y_scaled]).T
    
    if bandwidth is None:
        N_xy = len(xy_scaled)
        if N_xy > N_max:
            cv_select = np.random.choice(N_xy,N_max,replace=False)
            xy_scaled_cv = xy_scaled[cv_select]
        else:
            xy_scaled_cv = xy_scaled.copy()
        # calculate the bandwidth in 3 'scales'- coarse, fine, and hyperfine. This
        # reduces the total number of iterations.
        coarse_bandwidths = np.logspace(-2,0,10)
        _, coarse_bandwidth_range = find_best_bandwidth(xy_scaled_cv,coarse_bandwidths,
                                                        n_folds)
        fine_bandwidths = np.linspace(coarse_bandwidth_range[0],
                                      coarse_bandwidth_range[1],8)
        _, fine_bandwidth_range = find_best_bandwidth(xy_scaled_cv,fine_bandwidths,
                                                      n_folds)
        hyperfine_bandwidths = np.linspace(fine_bandwidth_range[0],
                                           fine_bandwidth_range[1],8)
        best_bandwidth, _ = find_best_bandwidth(xy_scaled_cv,hyperfine_bandwidths,
                                                n_folds)

    H, V, x_grid_scaled, y_grid_scaled, bandwidth = xy_kde(xy_scaled,best_bandwidth,
                                                           levels=levels) 
    x_grid = unscale_data(x_grid_scaled,x_mean,x_std)
    y_grid = unscale_data(y_grid_scaled,y_mean,y_std)
    if plot is True:
        plot_contour(x_grid,y_grid,H,V,fill,zorder,
                     line_properties,fill_properties)
  
    return x_grid, y_grid, H, V, bandwidth