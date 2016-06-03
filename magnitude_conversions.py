import numpy as np
import math

# M_sun = 4.74(e.g., Bessell, Castelli, & Plez 1998; Cox 2000; Torres 2010)

def Mag_to_lum(Mag,M_sun=4.74):
    # abs. mag --> solar lum.
    lum = 10**((M_sun-Mag)/2.5)
    return lum
  

def lum_to_Mag(lum,M_sun=4.76):
    # solar lum. --> abs. mag
    Mag = M_sun - 2.5*(np.log10(lum))
    return Mag