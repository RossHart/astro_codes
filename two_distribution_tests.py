import math
import numpy as np


def p_hist(data,bins):
    data_binned, _ =  np.histogram(data,bins)
    return data_binned/data_binned.sum()

def kullback_liebler(a,b,x_range=None,bins=10):
    if x_range is None:
        x_range = (np.min((a.min(),b.min())),
                   np.max((a.max(),b.max())))
    
    if type(bins) is int:
        bins = np.linspace(*x_range,num=bins)
    
    p_a = p_hist(a,bins)
    p_b = p_hist(b,bins)
    D_kl = 0
    for p_a_i, p_b_i in zip(p_a,p_b):
        D_kl += p_a_i*math.log10(p_a_i/p_b_i) if \
                (p_a_i > 0) & (p_b_i > 0) else 0
        
    return D_kl

    
def bhattacharyya_coefficient(a,b,x_range=None,bins=100):
    '''
    This measures the Bhattacharyya coefficient and Bhattacharyya distance 
    between two datasets, a and b.
  
    Inputs
    ------
    a: First distribution, numpy array or list of values
  
    b: second, distribution, same format as a.
    
    x_range: if None, then the range is set from the entire range of the 
    dataset. Can also set this as a two-object tuple, eg. (0,1)
    
    bins: if an integer, then this is the number of bins. Can also pass a list
    or an array to set the bin boundaries manually.
    
    Returns
    -------
    Bhattacharyya coefficent (BC): measure of distribution overlap.
    
    Bhattacharyya distance (D_B): measure of distance between the distributions.
    '''
  
    if x_range is None:
        x_range = (min((np.min(a),np.min(b))),
                   max((np.max(a),np.max(b))))
    
    if type(bins) is int:
        bins = np.linspace(*x_range,num=bins)
        
    p_a = p_hist(a,bins)
    p_b = p_hist(b,bins)
    BC = 0
    for p_ai, p_bi in zip(p_a,p_b):
        BC += math.sqrt(p_ai*p_bi)
    D_B = -math.log(BC)
    return BC, D_B