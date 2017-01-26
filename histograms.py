from astroML.density_estimation import knuth_bin_width
from astropy.table import Table, column
import math
import matplotlib.pyplot as plt
import numpy as np
from masking import select_within_range

from find_nearest import find_nearest_value, round_to_1sf
from contours import set_line_properties

def set_histogram_fill_properties(fill_properties=None):
    
    fp_final = {'color':'k',
                'alpha':0.4}  

    if fill_properties is not None:
        for f in fill_properties.keys():
            fp_final[f] = fill_properties[f]
    
    return fp_final


def get_knuth_bins(x,bounds=None):
  
    ''' 
    Code for finding the 'ideal' bin width for plotting histograms. The 'best'
    width is then converted to the the nearest 'OK' value (ie. a multiple of 
    0.1, 0.25, 0.5 or 1)
    
    Inputs:
    -------
    x: data that we wish to bin.
    
    bounds: upper and lower bounds to the data. If None, then the limits of the
    x data are used. Default is None.
    
    Returns:
    --------
    dx: optimal bin width
    
    bins: optimal bins for x.
    '''
    
    ok_widths = (0.1,0.25,0.5,1)
    
    if bounds is not None:
        x_mask = (x >= bounds[0]) & (x <= bounds[1])
        x_plot = x[x_mask]
    else:
        x_plot = x.copy()
        bounds = (np.min(x_plot),np.max(x_plot))
    dx = knuth_bin_width(x_plot)
    n_zeros = math.floor(math.log10(dx))
    ok_widths_modified = [w*10**(n_zeros+1) for w in ok_widths]
    dx_modified = find_nearest_value(ok_widths_modified,dx)
    bin_range = (dx_modified*(math.floor(bounds[0]/dx_modified)),
                 dx_modified*(math.ceil(bounds[1]/dx_modified)))
    bins = np.arange(bin_range[0],bin_range[1]+dx_modified*0.01,
                     dx_modified)
    
    return dx_modified, bins


def histogram(x,bins='knuth',x_range=None,fill=False,
              line_properties=None,fill_properties=None,
              normalise=True,weights=None,zorder=0,
              orientation='vertical'):
    
    _, x_plot, x_range = select_within_range(x,x_range)
    
    if bins is 'knuth':
        _, bins = get_knuth_bins(x_plot)
    elif isinstance(bins, int) is True:
        bins = np.linspace(x_range[0],x_range[1],bins)
    elif isinstance(bins, (list,np.ndarray)) is True:
        bins = bins
    
    fp = set_histogram_fill_properties(fill_properties)
    lp = set_line_properties(line_properties)
    if fill is True:
        _ = plt.hist(x,bins,normed=normalise,weights=weights,
                     histtype='stepfilled',alpha=fp['alpha'],
                     color=fp['color'],zorder=zorder,orientation=orientation)
    _ = plt.hist(x,bins,histtype='step',normed=normalise,weights=weights,
                 linewidth=lp['linewidth'],linestyle=lp['linestyle'],
                 color=lp['color'],alpha=lp['alpha'],zorder=zorder,orientation=orientation)
    
    return bins