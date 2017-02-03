import math
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.kernel_ridge import KernelRidge

from find_nearest import find_nearest_index
from masking import select_within_range


def scale_data(data):
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_scaled = (data-data_mean)/data_std
    return data_scaled, data_mean, data_std


def unscale_data(data_scaled,data_mean,data_std):
    data_unscaled = data_scaled*data_std + data_mean
    return data_unscaled
  

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


def xyz_contour(x,y,z,x_range=None,y_range=None,
                N_max=1000,n_folds=3,gamma=None,
                zorder=0,line_properties=None,
                plot=True,fill_properties=None):
    
    np.random.seed(0)

    select_x, _, x_range = select_within_range(x,x_range)
    select_y, _, y_range = select_within_range(y,y_range)
    select_xy = (select_x) & (select_y)
    xyz_plot = (np.array([x,y,z]).T)[select_xy]
    x_scaled, x_mean, x_std = scale_data(xyz_plot[:,0])
    y_scaled, y_mean, y_std = scale_data(xyz_plot[:,1])
    xyz_scaled = np.array([x_scaled,y_scaled,z]).T
    
    if gamma is None:
        N_xyz = len(xyz_scaled)
        if N_xyz > N_max:
            cv_select = np.random.choice(N_xyz,N_max,replace=False)
            xyz_scaled_cv = xy_scaled[cv_select]
        else:
            xyz_scaled_cv = xyz_scaled.copy()
            
        coarse_gammas = np.logspace(-2,0,10)
        _, coarse_gamma_range = find_best_gamma(xyz_scaled_cv,coarse_gammas,
                                                n_folds)
    
        fine_gammas = np.linspace(coarse_gamma_range[0],
                                  coarse_gamma_range[1],8)
        _, fine_gamma_range = find_best_gamma(xyz_scaled_cv,fine_gammas,
                                              n_folds)
    
        hyperfine_gammas = np.linspace(fine_gamma_range[0],
                                       fine_gamma_range[1],8)
        best_gamma, _ = find_best_gamma(xyz_scaled_cv,hyperfine_gammas,
                                        n_folds)
        
    H, x_grid_scaled, y_grid_scaled, bandwidth = xyz_kde(xyz_scaled,best_gamma)
    x_grid = unscale_data(x_grid_scaled,x_mean,x_std)
    y_grid = unscale_data(y_grid_scaled,y_mean,y_std)
    V = np.linspace(np.min(H),np.max(H),100)
    
    if plot is True:
        plot_contour(x_grid,y_grid,H,V,fill=True,line=False,
                     zorder=zorder,line_properties=line_properties,
                     fill_properties=fill_properties)
    
    return x_grid, y_grid, H, V


def plot_contour(x_grid,y_grid,H,V,fill=False,line=True,zorder=0,
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
        f = plt.contourf(x_grid,y_grid,H,levels=np.append(V,np.max(H))
                     ,cmap=fp['colormap'],alpha=fp['alpha'],zorder=zorder)
        #plt.colorbar(f)
    if line is True:
        plt.contour(x_grid,y_grid,H,levels=V,linewidths=lp['linewidth'],colors=lp['color'],
                    linestyles=lp['linestyle'],alpha=lp['alpha'],zorder=zorder)
    
    
    return None


def xyz_kde(xyz,gamma,N_grid=100):
    xy = xyz[:,:-1]
    z = xyz[:,-1]
    
    x_edges = np.linspace(np.min(xy[:,0]),np.max(xy[:,0]),N_grid+1)
    y_edges = np.linspace(np.min(xy[:,1]),np.max(xy[:,1]),N_grid+1)
    x_centres = np.array([x_edges[b] + (x_edges[b+1]-x_edges[b])/2 
                          for b in range(N_grid)])
    y_centres = np.array([y_edges[b] + (y_edges[b+1]-y_edges[b])/2 
                          for b in range(N_grid)])
    x_grid, y_grid = np.meshgrid(x_centres,y_centres)
    xy_grid = np.array([np.ravel(x_grid),np.ravel(y_grid)]).T
    clf = KernelRidge(kernel='rbf',gamma=gamma).fit(xy,z)
    H = clf.predict(xy_grid).reshape(N_grid,N_grid)
    return H, x_grid, y_grid, gamma


def cross_validate(xyz,gammas,n_folds=5):
    params = {'gamma':gammas}
    x_ = xyz[:,:-1]
    y_ = xyz[:,-1]
    kf = KFold(n_splits=n_folds,random_state=0)
    grid = GridSearchCV(KernelRidge(kernel='rbf'),params,cv=kf)
    grid.fit(x_, y_)
    return grid.best_estimator_.gamma, grid


def find_best_gamma(xyz,gammas,n_folds=5):
    N_gammas = len(gammas)
    best_gamma, _ = cross_validate(xyz,gammas,n_folds)
    i_best = find_nearest_index(gammas,best_gamma)
    if i_best == 0:
        gamma_lower_bound = gammas[0] - (gammas[1]-gammas[0])
    else:
        gamma_lower_bound = gammas[i_best-1]  
    if i_best == N_gammas-1:
        gamma_upper_bound = gammas[-1] + (gammas[-1]-gammas[-2])
    else:
        gamma_upper_bound = gammas[i_best+1]
    gamma_range = (gamma_lower_bound,gamma_upper_bound)
    return best_gamma, gamma_range