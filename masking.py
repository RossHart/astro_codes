from astropy.table import Table, column
import math
import numpy as np

def select_within_range(data,limits=None):
    if limits is None:
        select_data = np.isfinite(data)
        limits = (np.min(data),np.max(data))
    else:
        select_data = (data >= limits[0]) & (data <= limits[1])
    data_modified = data[select_data]
    return select_data, data_modified, limits