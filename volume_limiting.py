from astropy.table import Table
import math
import numpy as np
from calc_kcor import calc_kcor
from astropy.cosmology import FlatLambdaCDM
cosmo=FlatLambdaCDM(H0=70,Om0=0.3) # Use standard cosmology model from astropy.


def get_sample_limits(z_limit,mag_limit,color=0.75):
    '''
    ---Given a redshift and a magnitude limit, calculate an
    absolute magnitude limit---
    
    Inputs:
    -------
    z_limit: redshift limit of the volume-limited sample that we wish to define.
    
    mag_limit: apparent magnitude limit of the sample (eg. 17.0 for normal depth SDSS).
    
    color: g-r colour to calculate k-corrections.
    
    Outputs:
    --------
    M_limit: absolute magnitude limit.
    '''

    D_max = (10**6)*(np.array(cosmo.luminosity_distance([z_limit]))[0])
    #k_val = calc_kcor('r',z_limit,'g - r',color) # value of 0.75 is arbitrarily selected here!
    M_limit = mag_limit - 5*(math.log10(D_max) - 1) #- k_val

    return M_limit


def output_vls_sizes(data,z_range,mag_limit,column_names=['z','M_r']
                     ,N_steps=1000,low_z_limit=None,return_table=False):
    '''
    Cycle through a set of redshifts to find the 'best' redshift limit.
    
    Inputs:
    -------
    data: dataset we want to volume limit
    
    z_range: range of redshifts to 'cycle' through (eg. [0.03,0.1])
    
    mag_limit: apparent magnitude limit of the sample (eg. 17.0 for normal depth SDSS).
    
    column_names: 2 item list of column names (redshift name and Magnitude name).
    
    N_steps: number of steps to cycle through between the z_limits. Arbitrarily N=1000.
    
    low_z_limit: low redshift limit to the data. If None, no limit is applied. 
    Arbitrarily set as None.
    
    return_table: if False, only the bset paramaters are returned. If True, all data
    is returned.
    
    Outputs:
    --------
    best_params: dictionary with the keys 'redshift', 'Magnitude' and 'N_gal'.
    
    *N_table: if return_table is set as True, then a table with the columns 
    'redshift', 'Magnitude' and 'N_gal' is returned.
    '''
    
    z_name, M_name = column_names
    
    if low_z_limit != None:
        cut_z = data[z_name] >= low_z_limit
        data = data[cut_z]
    
    z_values = np.linspace(z_range[0],z_range[1],N_steps)
    N_array = np.zeros((N_steps,3))

    for i,z_limit in enumerate(z_values):
        M_limit = get_sample_limits(z_limit,mag_limit)
        in_volume = (data[z_name] <= z_limit) & (data[M_name] <= M_limit)
        N = np.sum(in_volume)
        N_array[i] = [z_limit,M_limit,N]
        
    max_N_row = np.argmax(N_array[:,2])
    best_params = {}
    best_params['redshift'] = N_array[max_N_row,0]
    best_params['Magnitude'] = N_array[max_N_row,1]
    best_params['N_gal'] = N_array[max_N_row,2]
    
    if return_table:
        N_table = Table(N_array,names=('redshift','Magnitude','N_gal'))
        return best_params,N_table
    else:
        return best_params


def get_volume_limited_sample(data,z_limit,mag_limit,column_names=['z','M_r'],
                              low_z_limit=None,append_column=True):
    
    '''
    --- Get a volume-limited sample, given a dataset and limits ---
    
    Inputs:
    -------
    data: dataset containing the Magnitude and redshift data.
    
    z_limit: upper limit on the redshift.
    
    mag_limit: apparent magnitude limit of the sample (eg. 17.0 for normal depth SDSS).
    
    column_names: 2 item list of column names (redshift name and Magnitude name).
    
    low_z_limit: low redshift limit to the data. If None, no limit is applied.
    
    append_column: if set as True, then a boolean column is added to the dataset
    (True if in volume limit, False if not).
    
    Outputs:
    --------
    *in_volume_limit: boolean array (True if in volume limit, False if not). Only
    returned if append_column = False
    
    *data: input array, but with an extra column added, called 'in_volume_limit'.
    (see above).
    Arbitrarily set as None.
    '''
    
    z_name, M_name = column_names
    
    if low_z_limit != None:
        in_z_min = data[z_name] >= low_z_limit
    else:
        in_z_min = np.full(len(data), True, dtype=bool)
    
    Mag_limit = get_sample_limits(z_limit,mag_limit)
    in_z_max = data[z_name] <= z_limit
    in_Mag = data[M_name] <= Mag_limit
    in_volume_limit = (in_z_min) & (in_z_max) & (in_Mag)
    
    if append_column:
        data['in_volume_limit'] = in_volume_limit
        return data
    else:
        return in_volume_limit