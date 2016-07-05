import numpy as np
import math
import astropy.units as u

# all paramaters obtained from Salim+2007 (SDSS SFR indicator)
  
def Mag_to_lum(Mag):
    S = 3631*10**(Mag/-2.5)*u.Jy # AB -> flux density
    L = S*(4*math.pi)*(10*u.pc)**2 # absolute magnitude = 10pc
    return L.to(u.erg/u.s/u.Hz)


def lum_to_sfr(L):
    return np.log10(1.08e-28*L.value) # conversion factor (Salim+2007) 


def modify_magnitude(FUV,NUV,r):    
    A_FUV = np.zeros(len(FUV))
    red = NUV - r < 4
    blue = red == False
    A_FUV[red] = red_convert(FUV[red],NUV[red])
    A_FUV[blue] = blue_convert(FUV[blue],NUV[blue])
    return A_FUV

    
def red_convert(FUV,NUV):
    A_FUV = np.zeros(len(FUV))
    red =  FUV - NUV < 0.95
    blue = red == False
    A_FUV[red] = 3.32*(FUV[red]-NUV[red]) + 0.22
    A_FUV[blue] = 3.37
    return A_FUV


def blue_convert(FUV,NUV):
    A_FUV = np.zeros(len(FUV))
    red =  FUV - NUV < 0.9
    blue = red == False
    A_FUV[red] = 2.99*(FUV[red]-NUV[red]) + 0.27
    A_FUV[blue] = 2.96
    return A_FUV
  

def FUV_to_SFR(FUV,NUV,r):
    '''
    Function for correcting and converting GALEX data to SFRs
    
    Inputs:
    -------
    FUV, NUV, r: AB magnitudes for the GALEX and SDSS data.
    
    Outputs:
    --------
    SFR: Star-formation rate array.
    
    L: _corrected_ FUV band luminosity.
    '''
    A_FUV = modify_magnitude(FUV,NUV,r) # dust corrected FUV.
    FUV_corrected = FUV - A_FUV
    L = Mag_to_lum(FUV_corrected)
    SFR = lum_to_sfr(L)
    return SFR, L
