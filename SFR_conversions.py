# Code for converting various luminosities or magnitudes to SFRs via simple conversions

import numpy as np
import math
import astropy.units as u
import copy

def L_22_to_SFR(L_22,eta=0.17,input_log=True,output_log=True):
  
  if input_log is True:
      L_22 = 10**L_22
  
  sfr = (1-eta)*(10**(-9.125))*L_22
  
  if output_log is True:
      return np.log10(sfr)
  else:
      return sfr
    

def L_FUV_to_SFR(L_FUV,input_log=True,output_log=True):
  
  if input_log is True:
      L_FUV = 10**L_FUV
  
  sfr = (10**(-9.69))*L_FUV
  sfr = sfr*(10**(-0.03)) # Kroupa -> Chabrier
  
  if output_log is True:
      return np.log10(sfr)
  else:
      return sfr
    


def L12_to_sfr(L12,log=True,snr=None):
    ''' 
    Function for converting luminosities to SFRs using the WISE 12 micron,
    with the equation provided by Chang+15.
    
    Inputs:
    -------
    L12: luminosity of the L12 band (in solar luminosities, as per Chang+15)
    
    log: if True, then the luminosities are in log units.
    
    snr: signal-to-noise (for error calculations)
    
    Outputs:
    --------
    log_sfr: log(SFR) in M*/yr
    
    *sigma_sfr: log of the error in log_sfr. Only returned if signal-to-noise
    is not None.
    '''
    if log == False:
        logL12 = np.log10(L12)
    else:
        logL12 = copy.copy(L12)
        L12 = 10**(L12)
    log_sfr = logL12 - 9.18
    if snr == None:
        return log_sfr
    else:
        sigma_sfr = np.log10(10**(-9.18)*(L12/snr)) + 0.2 # 0.2dex scatter in the relations.
        return log_sfr, sigma_sfr
    

def L22_to_sfr(L22,log=True,snr=None):
    ''' 
    Function for converting luminosities to SFRs using the WISE 12 micron,
    with the equation provided by Chang+15.
    
    Inputs:
    -------
    L22: luminosity of the L12 band (in solar luminosities, as per Chang+15)
    
    log: if True, then the luminosities are in log units.
    
    snr: signal-to-noise (for error calculations)
    
    Outputs:
    --------
    log_sfr: log(SFR) in M*/yr
    
    *sigma_sfr: log of the error in log_sfr. Only returned if signal-to-noise
    is not None.
    '''
    if log == False:
        logL22 = np.log10(L22)
    else:
        logL22 = copy.copy(L22)
        L12 = 10**(L22)
    log_sfr = logL22 - 9.08
    if snr == None:
        return log_sfr
    else:
        sigma_sfr = np.log10(10**(-9.08)*(L22/snr)) + 0.2 # 0.2 dex scatter
        # in the relations.
        return log_sfr, sigma_sfr
  
  
def FUV_to_sfr(FUV,snr=None):
    '''
    Function for correcting and converting GALEX data to SFRs
    
    Inputs:
    -------
    FUV: AB magnitude for the GALEX FUV filter.
    
    snr: signal-to-noise (for error calculations)
    
    Outputs:
    --------
    SFR: Star-formation rates.
    '''
    L = Mag_to_lum(FUV)
    log_sfr = np.log10(1.08e-28*L.value)
    if snr == None:
        return log_sfr
    else:
        sigma_sfr = np.log10(np.log(10)*(L.value/snr)*(1.08e-28)) + 0.26
        # 0.26 dex scatter in the resulting sfrs.
        return log_sfr, sigma_sfr


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
  

def FUV_to_sfr_w_dust_correction(FUV,NUV,r,snr=None):
    '''
    Function for correcting and converting GALEX data to SFRs
    
    Inputs:
    -------
    FUV, NUV, r: AB magnitudes for the GALEX and SDSS data.
    
    snr: error in the FUV flux.
    
    Outputs:
    --------
    SFR: Star-formation rate array.
    
    L: _corrected_ FUV band luminosity.
    '''
    A_FUV = modify_magnitude(FUV,NUV,r) # dust corrected FUV.
    FUV_corrected = FUV - A_FUV
    if FUV_error == None:
        log_sfr = FUV_to_SFR(FUV_corrected)
        return log_sfr
    else:
        log_sfr, sigma_sfr = FUV_to_SFR(FUV_corrected,snr)
        return log_sfr, sigma_sfr
        
