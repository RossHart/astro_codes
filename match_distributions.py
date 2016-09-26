import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold

def cross_validate(test_data,bandwidths,n_folds=5):
    params = {'bandwidth': bandwidths}
    kf = KFold(n=len(test_data),n_folds=n_folds,shuffle=True,random_state=0)
    grid = GridSearchCV(KernelDensity(), params,cv=kf)
    grid.fit(test_data)
    return grid.best_estimator_.bandwidth,grid


def get_kde(x,x_range=None,bandwidth=None,
            n_folds=3,printout=False,N_max=1000):

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
        
    return kde, bandwidth


def match_distributions(data,reference_data):
    '''
    Inputs:
    ------
    data: data that will be matched to another dataset.
  
    reference_data: the dataset to which 'data' will be matched.
  
    Outputs:
    -------
    mask: mask of length len(data), that will create the 
    matched distribution.
    
    p: probability for each of the points in 'data'.
    '''
    
    N_data, N_reference = [len(data),len(reference_data)]
    kde_reference, bandwidth = get_kde(reference_data)
    kde_data, _ = get_kde(data,bandwidth=bandwidth)
    data_ = data[:,np.newaxis]
    p_data = np.exp(kde_data.score_samples(data_))
    p_reference = np.exp(kde_reference.score_samples(data_))
    p = p_reference/p_data
    p_95 = np.percentile(p,95) # take the 95th percentile, to remove any outliers
    p = p/p_95
    p[p > 1] = 1
    mask = np.full(N_data,False,dtype=bool)
    for i, d in enumerate(data):
        mask[i] = np.random.binomial(1,p[i])
    
    return mask, p