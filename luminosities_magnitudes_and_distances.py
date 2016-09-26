from astropy.table import Table, column
import math
import numpy as np
import astropy.units as u
import astropy.constants as const


def z_to_dist(z):
    c = const.c
    H_0 = 70*(u.km/u.s/u.Mpc)
    D = (c/H_0*z).to(u.Mpc)
    return D


def dist_to_z(D):
    c = const.c
    H_0 = 70*(u.km/u.s/u.Mpc)
    z = (H_0/c)*D
    return z.to(u.dimensionless_unscaled)


def mag_to_Mag(mag,z):
    D = z_to_dist(z)
    Mag = mag - 5*(np.log10(D/(u.pc))-1)
    return Mag


def Mag_to_mag(Mag,z):
    D = z_to_dist(z)
    Mag = Mag + 5*(np.log10(D/(u.pc))-1)
    return Mag

    
def Mag_to_flux_density(Mag):
    S = 3631*10**(Mag/-2.5)*u.Jy # AB -> flux density
    L = S*(4*math.pi)*(10*u.pc)**2 # absolute magnitude = 10pc
    return L.to(u.erg/u.s/u.Hz)


def wavelength_to_frequency(wavelength):
    c = const.c
    frequency = (c/wavelength).to(u.Hz)
    return frequency


def Mag_to_lum(Mag,wavelength):
    frequency = wavelength_to_frequency(wavelength*(u.Angstrom))
    L_density = Mag_to_flux_density(Mag)
    L = L_density*frequency
    return L


def lum_to_solar(L):
    L_solar = 3.828e33*(u.erg/u.s)
    logLsun = np.log10(L/L_solar)
    return logLsun


def solar_to_lum(logLsun):
    L_solar = 3.828e33*(u.erg/u.s)
    L = 10**logLsun*L_solar
    return L


def lum_to_Mag(L,wavelength):
    c = const.c
    frequency = wavelength_to_frequency(wavelength*(u.Angstrom))
    L_density = (L/frequency).to(u.erg/u.s/u.Hz)
    S = (L_density/(4*math.pi*(10*u.pc)**2)).to(u.Jy)
    Mag = -2.5*np.log10(S/(3631*(u.Jy)))
    return Mag