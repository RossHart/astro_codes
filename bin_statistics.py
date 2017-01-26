from astropy.table import Table, column
import numpy as np
import math
import scipy.stats.distributions as dist

def assign_bins(x,x_range=None,equal_N=True,N_bins=10):
    if x_range is None:
        x_range = (np.min(x),np.max(x))
    
    in_range = (x >= x_range[0]) & (x <= x_range[1])
    
    if equal_N is False:
        bin_edges = np.linspace(x_range[0],x_range[1]+1,N_bins+1)
        bin_assignments = np.digitize(x,bin_edges)
    else:
        N_x = len(x)
        bin_edges = np.linspace(0,1,N_bins+1)
        bin_edges[-1] += 1
        v_values = np.linspace(0,1,N_x)
        v = np.zeros(N_x)
        for i, x_ in enumerate(np.argsort(x)):
            v[x_] = v_values[i]
        bin_assignments = np.digitize(v,bin_edges)
    bin_assignments[in_range == False] = -999
    
    return bin_assignments


def get_fractional_errors(k,n,c=0.683):
    p_lower = dist.beta.ppf((1-c)/2.,k+1,n-k+1)
    p_upper = dist.beta.ppf(1-(1-c)/2.,k+1,n-k+1)
    return p_lower,p_upper


class stats_functions:
        
    def __init__(self,percentiles=(16,84),sigmas=(1,2),c=(0.683)):
        self.percentiles = ([percentiles] if isinstance(percentiles,int) | isinstance(percentiles,float) 
                            else percentiles)
        self.sigmas = ([sigmas] if isinstance(sigmas,int) | isinstance(sigmas,float) 
                       else sigmas)
        self.c = ([c] if isinstance(c,int) | isinstance(c,float) 
                  else c)

    def median_and_percentile(self,data,bins):
        output_table = Table()
        N_bins = np.unique(bins)
        N_bins = N_bins[N_bins > 0]
        medians = [np.median(data[bins == b]) for b in N_bins]
        maxs = [np.max(data[bins == b]) for b in N_bins]
        mins = [np.min(data[bins == b]) for b in N_bins]
        means = [np.mean(data[bins == b]) for b in N_bins]
        standard_deviations = [np.mean(data[bins == b]) for b in N_bins]
        standard_errors = [np.mean(data[bins == b]) for b in N_bins]
        output_table['median'] = medians
        output_table['max'] = maxs
        output_table['min'] = mins
        for p in self.percentiles:
            percentiles = [np.percentile(data[bins == b],p) for b in N_bins]
            output_table['{} percentile'.format(p)] = percentiles
        return output_table


    def mean_and_deviation(self,data,bins):
        output_table = Table()
        N_bins = np.unique(bins)
        N_bins = N_bins[N_bins > 0]
        means = [np.mean(data[bins == b]) for b in N_bins]
        maxs = [np.max(data[bins == b]) for b in N_bins]
        mins = [np.min(data[bins == b]) for b in N_bins]
        output_table['mean'] = means
        output_table['max'] = maxs
        output_table['min'] = mins
        standard_deviations = [np.mean(data[bins == b]) for b in N_bins]
        for s in self.sigmas:
            output_table['mean-{}sigma'.format(s)] = [means[i]-s*standard_deviations[i] 
                                                      for i, _ in enumerate(means)]
            output_table['mean+{}sigma'.format(s)] = [means[i]+s*standard_deviations[i] 
                                                      for i, _ in enumerate(means)]
        return output_table
        
        
    def mean_and_error(self,data,bins):
        output_table = Table()
        N_bins = np.unique(bins)
        N_bins = N_bins[N_bins > 0]
        means = [np.mean(data[bins == b]) for b in N_bins]
        maxs = [np.max(data[bins == b]) for b in N_bins]
        mins = [np.min(data[bins == b]) for b in N_bins]
        output_table['mean'] = means
        output_table['max'] = maxs
        output_table['min'] = mins
        standard_errors = [np.std(data[bins == b])/np.sqrt(np.sum(bins == b)) 
			   for b in N_bins]
        for s in self.sigmas:
            output_table['mean-{}sigma'.format(s)] = [means[i]-s*standard_errors[i] 
                                                      for i, _ in enumerate(means)]
            output_table['mean+{}sigma'.format(s)] = [means[i]+s*standard_errors[i] 
                                                      for i, _ in enumerate(means)]
        return output_table
    
    
    def fraction_with_feature(self,data,bins):
        output_table = Table()
        N_bins = np.unique(bins)
        N_bins = N_bins[N_bins > 0]
        n = np.array([np.sum(bins == b) for b in N_bins])
        k = np.array([np.sum(data[bins == b]) for b in N_bins])
        output_table['f'] =k/n
        for c_ in self.c:
            p_lower, p_upper = get_fractional_errors(k,n,c_)
            output_table['f-{}'.format(c_)] = p_lower
            output_table['f+{}'.format(c_)] = p_upper
        return output_table