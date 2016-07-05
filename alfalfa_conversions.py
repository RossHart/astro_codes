import math
import numpy as np

def mass_from_redshift(z):
    '''
    From a given redshift, calculate the mass of HI above which the sample 
    is complete.
    '''
    K = 2.356e5*((3e5/70)**2) # constant for converting between flux and mass
    mass_limit = np.log10(K*(z**2)*0.72)
    return mass_limit


def redshift_from_distance(distance):
    '''
    From a value of distance, calculate the corersponding redshift. 
    *this only works for z>=0.02*
    '''
    return (70/3e5)*distance
  
  
def distance_from_redshift(z):
    '''
    From a value of redshift, calculate the corersponding distance. 
    *this only works for z>=0.02*
    '''
    return (3e5/70)*z