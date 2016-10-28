import numpy as np

def find_nearest_value(array,value):
    i = (np.abs(array-value)).argmin()
    return array[i]
  
  
def find_nearest_index(array,value):
    i = (np.abs(array-value)).argmin()
    return i
  
  
def interpolate_for_y(x,y,value):
    '''
    Function for linear interpolation between x points, to find a corresponding y value
    for a given x value.
    
    Inputs:
    -------
    x: array of points
    
    y: N-dimensional array, same length as x, but as many columns as there are variables
    
    Outputs:
    --------
    y_interpolated: y, or list of N y values.
    '''
    x_overwrite = x.copy()
    x1v = find_nearest_index(x_overwrite,value)
    x_overwrite[x1v] = -np.inf
    x2v = find_nearest_index(x_overwrite,value)
    x1, x2 = x[x1v], x[x2v]
    y1, y2 = y[x1v], y[x2v]
    tan_alpha = (y1-y2)/(x1-x2)
    x_ = value - x1
    y_ = x_*tan_alpha
    y_interpolated = y1 + y_
    return y_interpolated
  
  
def round_to_1sf(v):
    return round(v, -int(math.floor(math.log10(abs(v)))))