# --------------------------------------------------------
#      Importing data dict from EN1 fits
# --------------------------------------------------------
def data_import(fits_fname='', pickle_name=''):    
    from astropy.io import fits
    import numpy as np
    import os, pickle

    fitshdu = fits.open(fits_fname)
    fitsarr = fitshdu[1].data
    fitscols = fitshdu[1].columns
    fitskeys = {fitscols[ii].name:ii for ii in range(len(fitscols)) }

    fluxkeys = np.array([fitskeys[ff] for ff in fitskeys if '_f' in ff])
    flux_e_keys = fluxkeys +1
    
    if pickle_name != '' and os.path.isfile(pickle_name):
        with open(pickle_name, 'rb') as handle:
            fitsdict = pickle.load(handle)
    else:
        fitsdict = {}
        cnt = 0
        for ths in fitsarr:
            if (np.isfinite(ths[fitskeys['Z_BEST']]))  and (ths[fitskeys['ch1_swire_f']] > 10e-6):
                fitsdict[cnt] = ( {
                    'sour_id' : str(ths[fitskeys['ID']]),
                    'z_median': ths[fitskeys['z1_median']],
                    'z_best' : ths[fitskeys['Z_BEST']],
                    'good_fluxes' : ths[fitskeys['Nband']],
                    'flux_maggies' : np.array([ths[jj]/3631.  for jj in fluxkeys]),
                    'flux_err_maggies' : np.array([ths[jj]/3631.  for jj in flux_e_keys])
                } )
                cnt+=1

    filterlist = np.array([ff[:-2] for ff in fitskeys if '_f' in ff])
    return fitsdict, filterlist
    
# --------------------------------------------------------
#      Query filters from sedpy
# --------------------------------------------------------

def get_filters(FILT_DIR = '', filts_from_file = [], **kwargs):
    import sedpy
    """
    Need to make it better instead of just hammering through it
    """
    filt_names = ['megacam_u','phaskell_gpc1_g', 'phaskell_gpc1_r', 'phaskell_gpc1_i',
    'phaskell_gpc1_z', 'phaskell_gpc1_y',  'phaskell_suprime_g', 'phaskell_suprime_r', 
    'phaskell_suprime_i', 'phaskell_suprime_z', 'phaskell_suprime_y', 'phaskell_suprime_n921', 
    'phaskell_ukidss_j', 'phaskell_ukidss_k', 'phaskell_irac_i1', 'phaskell_irac_i2', 
    'phaskell_irac_i3', 'phaskell_irac_i4', 'phaskell_irac_i1', 'phaskell_irac_i2', 
    'phaskell_mips_24', 'phaskell_pacs_green_100', 'phaskell_pacs_red_160', 
    'phaskell_spire_250', 'phaskell_spire_350', 'phaskell_spire_500']
    return sedpy.observate.load_filters(filt_names, directory=FILT_DIR)


def get_obj_by_radioid(obs_data,objradioid):
    tkey = -1
    for tt in obs_data.keys():
        if int(obs_data[tt]['radio_id']) == int(objradioid) :
            tkey = tt
            break
    return tkey