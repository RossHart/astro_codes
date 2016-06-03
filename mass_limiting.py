from astropy.table import Table
import math
import numpy as np
from calc_kcor import calc_kcor
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import curve_fit
from volume_limiting import get_sample_limits
cosmo=FlatLambdaCDM(H0=70,Om0=0.3) # Use standard cosmology model from astropy.
  
  
def baldry_equation(u_r):
    
    if u_r < 79/38:
        log_ML =  -0.95 + 0.56*(u_r)
    else:
        log_ML =  -0.16 + 0.18*(u_r)
    
    return log_ML


def Mag_to_lum(Mag,Mag_sun=4.67):
    return 10**((Mag_sun-Mag)/(2.5))


def log_function(x,a,b):
    return a*np.log10(x) + b


def get_mass_limits(data,z_range,N_steps,mag_limit,mass_equation,low_z_limit=None,column_names=['z','M_r'],
                    colours=['u','r'],Mag_sun=4.67):
    '''
    --- Given a dataset, and a method for converting colour to log(Mass/luminosity),
    return a set of stellar mass limits.---
    
    Inputs:
    -------
    data: dataset that we wish to stellar mass-limit
    
    z_range: limits we wish to explore
    
    N_steps: number of steps 
    
    mag_limit: apparent magnitude limit of the sample (eg. 17.0 for normal depth SDSS).
    
    mass_equation: equation for converting colour -> log(Mass/luminosity) (eg. Baldry+ 2006)
    
    low_z_limit: low redshift cut to apply
    
    column_names: redshift and Magnitude columns for the table.
    
    colours: colour1 and colour2 of the table (eg. u+r Magnitudes for Baldry+ 2006)
    
    Mag_sun: absolute magnitude of the sun (to get luminosity in solar luminosities). For 
    the SDSS r-band, this value is 4.67.
    
    Outputs:
    --------
    limit_table: table of reshifts with corresponding Magnitude and mass limits.
    
    fit_paramaters: best fit parameters of the form logM* = alog(z) + b 
    '''
    
    z_column, Mag_column = column_names
    colour = data[colours[0]] - data[colours[1]]
    
    if low_z_limit != None:
        in_z_min = data[z_column] >= low_z_limit
    else:
        in_z_min = np.full(len(data), True, dtype=bool)
    
    z_steps = np.linspace(z_range[0],z_range[1],N_steps)
    
    limit_array = np.zeros((N_steps,3))
    
    for i,z_limit in enumerate(z_steps):
        Mag_limit = get_sample_limits(z_limit,mag_limit)
        in_z_max = data[z_column] <= z_limit
        in_Mag_limit = data[Mag_column] <= Mag_limit
        in_volume_limit = (in_z_min) & (in_z_max) & (in_Mag_limit)
        vl_colour = colour[in_volume_limit]
        colour_99 = np.percentile(vl_colour,99,axis=0)
        log_ML = mass_equation(colour_99)
        lum_limit = Mag_to_lum(Mag_limit)
        mass_limit = lum_limit*(10**(log_ML))
        logmass = math.log10(mass_limit)
        limit_array[i] = [z_limit,Mag_limit,logmass]
    
    limit_table = Table(limit_array,names=('z','Mag','mass'))
    xy = np.array([limit_table['z'],limit_table['mass']]).T
    fit_paramaters, _ = curve_fit(log_function,xy[:,0],xy[:,1])
    return limit_table, fit_paramaters


def get_mass_limit(data,z_limit,mag_limit,mass_equation,low_z_limit=None,column_names=['z','M_r'],
                   colours=['u','r'],Mag_sun=4.67):
    '''
    --- Given a single luminosity limit, calculate the stellar mass-limit ---
    
    Inputs:
    -------
    data: dataset that we wish to stellar mass-limit
    
    z_limit: redshift limit of the volume-limited sample
    
    mag_limit: apparent magnitude limit of the sample (eg. 17.0 for normal depth SDSS).
    
    mass_equation: equation for converting colour -> log(Mass/luminosity) (eg. Baldry+ 2006)
    
    low_z_limit: low redshift cut to apply
    
    column_names: redshift and Magnitude columns for the table.
    
    colours: colour1 and colour2 of the table (eg. u+r Magnitudes for Baldry+ 2006)
    
    Mag_sun: absolute magnitude of the sun (to get luminosity in solar luminosities). For 
    the SDSS r-band, this value is 4.67.
    
    Outputs:
    --------
    limit_table: table of reshifts with corresponding Magnitude and mass limits.
    
    fit_paramaters: best fit parameters of the form logM* = alog(z) + b 
    '''
    
    z_column, Mag_column = column_names
    colour = data[colours[0]] - data[colours[1]]
    
    if low_z_limit != None:
        in_z_min = data[z_column] >= low_z_limit
    else:
        in_z_min = np.full(len(data), True, dtype=bool)
    
    Mag_limit = get_sample_limits(z_limit,mag_limit)
    in_z_max = data[z_column] <= z_limit
    in_Mag_limit = data[Mag_column] <= Mag_limit
    in_volume_limit = (in_z_min) & (in_z_max) & (in_Mag_limit)
    vl_colour = colour[in_volume_limit]
    colour_99 = np.percentile(vl_colour,99,axis=0)
    log_ML = mass_equation(colour_99)
    lum_limit = Mag_to_lum(Mag_limit)
    mass_limit = lum_limit*(10**(log_ML))
    logmass = math.log10(mass_limit)

    return logmass
  
# schechter parameters obtained from Baldry+2011
#def schechter_function(M,M_star=10**(10.66),phi1=3.96e-3,alpha1=-0.35,phi2=0.79e-3,alpha2=-1.47):
    #return (np.exp(-M/M_star))*((phi1*(M/M_star)**alpha1) + (phi2*(M/M_star)**alpha2))*(1/M_star)