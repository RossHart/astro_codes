from astropy.table import Table, column
import numpy as np
import math
import scipy.stats.distributions as dist

def assign_bins(x,x_range=None,equal_N=True,N_bins=10):
    if x_range is None:
        x_range = (np.min(x),np.max(x))
    
    in_range = (x >= x_range[0]) & (x <= x_range[1])
    
    if isinstance(N_bins,(list,np.ndarray)) is True:
        bin_assignments = np.digitize(x,N_bins)
        bin_assignments[bin_assignments < N_bins.min()] = -999
        bin_assignments[bin_assignments > N_bins.max()] = -999
    
    elif equal_N is False:
        bin_edges = np.linspace(x_range[0],x_range[1],N_bins+1)
        bin_edges[-1] += 1
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


class StatsFunctions():
    
    def __init__(self,data,weights=None):
        self.data = data
        self.weights = weights
        
    def percentiles(self,percentiles):
        data = self.data
        weights = self.weights
        if weights is None:
            return np.percentile(data, percentiles)
        else:
            indices = np.argsort(data)
            d_i = data[indices]
            w_i = weights[indices]
            p = w_i.cumsum()/w_i.sum()*100
            y = np.interp(percentiles, p, d_i)
            return y
    
    def mean_and_deviation(self):
        data = self.data
        weights = self.weights
        if weights is None:
            return np.mean(data), np.std(data)
        else:
            mean = np.average(data,weights=weights)
            variance = np.average((data-mean)**2,weights=weights)  # Fast and numerically precise
            return mean, math.sqrt(variance)
    
    def mean_and_error(self):
        data = self.data
        weights = self.weights
        N_scaler = np.sqrt(len(data))
        if weights is None:
            return np.mean(data), np.std(data)/N_scaler
        else:
            mean = np.average(data,weights=weights)
            variance = np.average((data-mean)**2,weights=weights)  # Fast and numerically precise
            return mean, math.sqrt(variance)/N_scaler


class TableStats():
    
    def __init__(self,data,bins,weights=None,c=0.683):
        self.data = data
        self.bins = bins
        self.weights = weights
        self.c = c
        
    def list_(self,values):
        values = ([values] if isinstance(values,int) | isinstance(values,float) 
                            else values)
        return values
    
    def median_and_percentile(self,percentiles=(16,84)):
        data = self.data
        bins = self.bins
        weights = self.weights
        percentiles = self.list_(percentiles)
        
        output_table = Table()
        N_bins = np.unique(bins)
        N_bins = N_bins[N_bins > 0]       
        
        if weights is None:
            median_list = [StatsFunctions(data[bins == b]).percentiles(50) 
                           for b in N_bins]
            output_table['mean'] = median_list
            for p in percentiles:
                percentile_list = [StatsFunctions(data[bins == b]).percentiles(p) 
                           for b in N_bins]
                output_table['{} percentile'.format(p)] = percentile_list
        
        else:
            median_list = [StatsFunctions(data[bins == b],weights[bins==b]).percentiles(50) 
                           for b in N_bins]
            output_table['mean'] = median_list
            for p in percentiles:
                percentile_list = [StatsFunctions(data[bins == b],weights[bins==b]).percentiles(p) 
                                   for b in N_bins]
                output_table['{} percentile'.format(p)] = percentile_list

        return output_table
      
    def median_and_error(self,sigmas=1):
        data = self.data
        bins = self.bins
        weights = self.weights
        sigmas = self.list_(sigmas)
        
        output_table = Table()
        N_bins = np.unique(bins)
        N_bins = N_bins[N_bins > 0]       
        
        if weights is None:
            median_list = [StatsFunctions(data[bins == b]).percentiles(50) 
                           for b in N_bins]
            output_table['mean'] = median_list
            meanstd_ = [StatsFunctions(data[bins == b]).mean_and_deviation() 
                        for b in N_bins]
        
        else:
            median_list = [StatsFunctions(data[bins == b],weights[bins == b]).percentiles(50) 
                           for b in N_bins]
            meanstd_ = [StatsFunctions(data[bins == b],weights[bins == b]).mean_and_deviation() 
                        for b in N_bins]
            output_table['mean'] = median_list
        
        N_array = np.array([(bins==b).sum() for b in N_bins])
        median_array = np.array(median_list)
        std_array = np.array([m_[1] for m_ in meanstd_])*1.253 # 1.25*error for median!
        error_array = std_array/np.sqrt(N_array)
        for s in sigmas:
            output_table['mean-{}sigma'.format(s)] = median_array - s*error_array
            output_table['mean+{}sigma'.format(s)] = median_array + s*error_array
      
        return output_table
    
    def mean_and_deviation(self,sigmas=1):
        data = self.data
        bins = self.bins
        weights = self.weights
        sigmas = self.list_(sigmas)
        
        output_table = Table()
        N_bins = np.unique(bins)
        N_bins = N_bins[N_bins > 0]
        
        if weights is None:
            meanstd_ = [StatsFunctions(data[bins == b]).mean_and_deviation() 
                        for b in N_bins]
        else:
            meanstd_ = [StatsFunctions(data[bins==b],weights[bins==b]).mean_and_deviation()
                        for b in N_bins]
        mean_array = np.array([m_[0] for m_ in meanstd_])
        std_array = np.array([m_[1] for m_ in meanstd_])
        
        output_table['mean'] = mean_array
        for s in sigmas:
            output_table['mean-{}sigma'.format(s)] = mean_array - s*std_array
            output_table['mean+{}sigma'.format(s)] = mean_array + s*std_array
        return output_table
    
    def mean_and_error(self,sigmas=1):
        data = self.data
        bins = self.bins
        weights = self.weights
        sigmas = self.list_(sigmas)
        
        output_table = Table()
        N_bins = np.unique(bins)
        N_bins = N_bins[N_bins > 0]
        
        if weights is None:
            meanstd_ = [StatsFunctions(data[bins == b]).mean_and_deviation() 
                        for b in N_bins]
        else:
            meanstd_ = [StatsFunctions(data[bins==b],weights[bins==b]).mean_and_deviation()
                        for b in N_bins]
        mean_array = np.array([m_[0] for m_ in meanstd_])
        std_array = np.array([m_[1] for m_ in meanstd_])
        N_array = np.array([(bins==b).sum() for b in N_bins])
        error_array = std_array/np.sqrt(N_array)
        
        output_table['mean'] = mean_array
        for s in sigmas:
            output_table['mean-{}sigma'.format(s)] = mean_array - s*error_array
            output_table['mean+{}sigma'.format(s)] = mean_array + s*error_array
        return output_table
    
    def fraction_with_feature(self):
        data = self.data
        bins = self.bins
        
        output_table = Table()
        N_bins = np.unique(bins)
        N_bins = N_bins[N_bins > 0]
        n = np.array([np.sum(bins == b) for b in N_bins])
        k = np.array([np.sum(data[bins == b]) for b in N_bins])
        output_table['mean'] =k/n
        for c_ in self.list_(self.c):
            p_lower, p_upper = get_fractional_errors(k,n,c_)
            output_table['mean-1sigma'] = p_lower
            output_table['mean+1sigma'] = p_upper
        return output_table