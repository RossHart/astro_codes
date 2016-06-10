import numpy as np
import math
import astropy.units as u

  
def FUV_Mag_to_luminosity(Mag):
    '''
    Function for converting from absolute magnitude -> luminosity.
    
    Inputs:
    -------
    Mag: absolute magnitude
    
    Outputs:
    --------
    L: luminosity (in ergs/s/Hz)
    '''
    
    K = 4*math.pi*(10*u.pc)**2 #4pi(D**2)
    S_Jy = 10**((27.5-Mag)/2.5)*10**(-6)*u.Jy # SDSS 'Pogson' magnitude
    L = K*S_Jy
    return L.to(u.erg/u.s/u.Hz)
  
