import os, sys
from tqdm.notebook import tqdm_notebook
import fitsio
from fitsio import FITS,FITSHDR
from astropy.io import fits
import numpy as np

def getlookback_fromz(z, z0=0):
    '''
    given redshift, returns lookbback time in yrs
    '''
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    return (cosmo.age(z0).value - cosmo.age(z).value )* 1e9 #lookback time in yrs

def gettage_fromz(z):
    '''
    given redshift, returns age of universe at that redshift in yrs
    '''
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    return ( cosmo.age(z).value )* 1e9 #lookback time in yrs

def getz_fromlookback(tl, z0=0):
    '''
    given lookback time in Gyrs, returns redshift
    '''
    from astropy.cosmology import z_at_value, FlatLambdaCDM
    import astropy.units as u
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    
    tage = cosmo.age(z0).value - tl
    return z_at_value(cosmo.age, tage * u.Gyr) 

def getz_fromage(tage):
    '''
    given age of universe in yrs, returns redshift
    '''
    from astropy.cosmology import z_at_value, FlatLambdaCDM
    import astropy.units as u
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    
    return z_at_value(cosmo.age, tage * u.yr, method='Bounded').value