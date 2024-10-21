import os, sys
from tqdm import tqdm
import fitsio
from fitsio import FITS,FITSHDR
from astropy.io import fits
import numpy as np


from functools import reduce

def allfactors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def mult_err(x,dx,y,dy, mode=None):
    '''
    z = x*y or x/y
    mode = mult/div. If mode undefined, then return just the sqrt part
    '''
    if mode=='mult':
        return abs(x*y)*( (dx/x)**2 + (dy/y)**2 )**0.5
    elif mode=='div':
        return abs(x/y)*( (dx/x)**2 + (dy/y)**2 )**0.5
    else:
        return ( (dx/x)**2 + (dy/y)**2 )**0.5

def type_str(var):
    return var.__class__.__name__

def ulen(x):
    try:
        if 'str' in type_str(x): return -1
        elif 'int' in type_str(x) or 'float' in type_str(x): return 0
        else: return len(x)
    except Exception as e:
        return -2    
    
# Calculate chi^2
def get_chi2(yobs,ymodel,yobs_err):
    ''' Calculate the chi^2 between observed (x) and 
    simulated (y) data given error in obs (yobs_err).
    Returns array of chi, chi^2, and mask array, where
    mask denotes the array indices where chi_arr is not null.
    return chi_arr, chi_sq, chi_nan_mask
    '''
    import numpy as np
    chi_arr, chi_nan_mask = [],[]

    # chi error for residual plot
    chi_arr = (yobs - ymodel)/yobs_err
    
    chi_nan_mask = np.array([ii for ii in range(len(chi_arr)) if np.isfinite(chi_arr[ii])])

    chi_sq_arr = np.array([ii**2 for ii in chi_arr[chi_nan_mask]])
    chi_sq = np.nansum(chi_sq_arr)

    return chi_arr, chi_sq, chi_nan_mask

# def loadfits2dict(fitsfile,colname=None):
#     fitshdu = fits.open(fitsfile)
#     fitsarr = fitsio.read(fitsfile)
#     fitscols = fitshdu[1].columns
#     fitskeys = {fitscols[ii].name:ii for ii in range(len(fitscols)) }
#     fitsdict = {}
#     if colname:
#         for ths in tqdm(fitsarr): fitsdict[int(ths[fitskeys[colname]])] = ths
#     else:
#         cnt = 0
#         for ths in tqdm(fitsarr): 
#             fitsdict[cnt] = ths
#             cnt+=1
#     for aa in list(fitsdict): 
#         if aa < 0: del fitsdict[aa]
#     return fitsdict, fitskeys

def loadfits2dict(fitsfile,colname=None,dotqdm=False):
    fitshdu = fits.open(fitsfile)
    fitsarr = fitshdu[1].data
    fitscols = fitshdu[1].columns
    fitskeys = {fitscols[ii].name:ii for ii in range(len(fitscols)) }
    fitsdict = {}
    if colname:
        if dotqdm:
            for ths in tqdm(fitsarr): fitsdict[int(ths[fitskeys[colname]])] = ths
        else:
            for ths in fitsarr: fitsdict[int(ths[fitskeys[colname]])] = ths
    else:
        cnt = 0
        if dotqdm:
            for ths in tqdm(fitsarr): 
                fitsdict[cnt] = ths
                cnt+=1
        else:
            for ths in fitsarr: 
                fitsdict[cnt] = ths
                cnt+=1

    for aa in list(fitsdict): 
        if aa < 0: del fitsdict[aa]
    return fitsdict, fitskeys

def arr_from_dict(tdict,tkeys,whichfield,troi=None,log=False):
    if troi is None: troi = list(tdict.keys())
    if log: return np.log10([tdict[xx][tkeys[whichfield]] for xx in troi])
    else: return np.array([tdict[xx][tkeys[whichfield]] for xx in troi])


def getvalwithsig(fdict,fdictkeys,fparam,common_keys,keydef=['_50','_16','_50','_84']):
    '''
    getvalwithsig(fdict,fdictkeys,fparam,common_keys)
    return fparamarr, fparamsig, fparamsigplus, fparamsigminus
    '''
    fparamarr = np.array([fdict[xx][fdictkeys['{0}{1}'.format(fparam,keydef[0])]] for xx in common_keys])
    fparamsigplus = np.array([fdict[xx][fdictkeys['{0}{1}'.format(fparam,keydef[3])]] 
                              - fdict[xx][fdictkeys['{0}{1}'.format(fparam,keydef[2])]] 
                        for xx in common_keys])
    fparamsigminus =  np.array([fdict[xx][fdictkeys['{0}{1}'.format(fparam,keydef[2])]] 
                                - fdict[xx][fdictkeys['{0}{1}'.format(fparam,keydef[1])]]
                        for xx in common_keys])
    fparamsig = np.array(list(zip(fparamsigminus, fparamsigplus))).T  
    return fparamarr, fparamsig, fparamsigplus, fparamsigminus

def stitch_errorbars(eup,edn):
    '''
    returns combined errorbars given the upper and lower errorbars
    '''
    import numpy as np
    return np.array(list(zip(edn, eup))).T     

def getlogwithsig(fdict,fdictkeys,fparam,common_keys,keydef=['_16','_50','_84']):
    '''
    getvalwithsig(fdict,fdictkeys,fparam,common_keys)
    return fparamarr, fparamsig, fparamsigplus, fparamsigminus
    '''
    fparamarr = np.log10([fdict[xx][fdictkeys['{0}{1}'.format(fparam,keydef[1])]] for xx in common_keys])
    fparamsigplus = np.log10([fdict[xx][fdictkeys['{0}{1}'.format(fparam,keydef[2])]] 
                             for xx in common_keys]) - fparamarr
    fparamsigminus =  fparamarr - np.log10([fdict[xx][fdictkeys['{0}{1}'.format(fparam,keydef[0])]] 
                                for xx in common_keys])
    fparamsig = np.array(list(zip(fparamsigminus, fparamsigplus))).T  
    return fparamarr, fparamsig, fparamsigplus, fparamsigminus

def log_ebar(x, xe):
    '''
    log_ebar(x, xe): convert the errors to log10 base
    return np.log10(x), logxe, logxeup, logxedn
    '''
    logxeup = np.log10(x+xe) - np.log10(x)
    logxedn = np.log10(x) - np.log10(x-xe)
    logxe = np.array(list(zip(logxedn, logxeup))).T
    
    return np.log10(x), logxe, logxeup, logxedn

def ezip(edn, eup):
    return np.array(list(zip(edn,eup))).T    

def getbagpmass(bagpdict, bagpkeys, common_keys):
    # bagppipes
    bagpmass = np.array([10**(bagpdict[xx][bagpkeys['stellar_mass_50']]) for xx in common_keys])
    bagpmasssigplus = np.array([ 
        10**(bagpdict[xx][bagpkeys['stellar_mass_84']]) - 10**(bagpdict[xx][bagpkeys['stellar_mass_50']])
                  for xx in common_keys])
    bagpmasssigminus = np.array([ 
        10**(bagpdict[xx][bagpkeys['stellar_mass_50']]) - 10**(bagpdict[xx][bagpkeys['stellar_mass_16']])
                  for xx in common_keys])

    bagpmasssig = np.array(list(zip(bagpmasssigminus,bagpmasssigplus))).T
    
    return bagpmass, bagpmasssig, bagpmasssigplus, bagpmasssigminus

def get_common_keys(*args,**kwargs):
    '''
    Get common elements from multiple arrays
    '''
    if(len(args)<2): return []
    common_keys = np.intersect1d(list(args[0]),list(args[1]))
    N = len(args)-2
    for ii in range(N):
        common_keys = np.intersect1d(common_keys,list(args[ii+2]))
    print('Common objects : ',len(common_keys))
    return common_keys

def get_uncommon_keys(*args,**kwargs):
    '''Return objects that are in only one array'''
    ak = np.unique(np.concatenate([*args]))
    ck = get_common_keys(*args)
    nk = np.array([xx for xx in ak if xx not in ck])
    print('Unique objs : ',len(nk))
    return nk    

# --------------------------------------------------------- #
#   Binmids
# --------------------------------------------------------- #
def get_binmids(xbins,edges=False):
    xmids = np.zeros(len(xbins)-1)
    for ii in range(len(xbins)-1):
        xmids[ii] = (xbins[ii]+xbins[ii+1])/2
            
    if edges:
        bdn, bup = xbins[0] - (xbins[1] - xbins[0]), xbins[-1] + (xbins[-1] - xbins[-2])
        return np.concatenate([[bdn],xmids,[bup]])
    else:
        return xmids

# --------------------------------------------------------- #
#   Quantiles
# --------------------------------------------------------- #
def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def peak_from_kde(x,bandwidth=1.0):
    import numpy as np
    from scipy.stats import norm
    from sklearn.neighbors import KernelDensity

    # Use kernel density estimator to estimate the PDF of the distribution
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x.reshape(-1, 1))
    pdf = np.exp(kde.score_samples(x.reshape(-1, 1)))

    # Find the peak of the PDF
    peak = x[np.argmax(pdf)]

    print(f"Peak: {peak:.3f}")
    return peak

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def straight_line(x,m,c): return m*x+c

def half_hist_to_gauss(tdata,tbins,peak=None,direction='left',edgetouse='upper',doplot=False,verbose=False):
    '''
    To return mean and std of a 1-d normal distribution with upper
    and lower bounds fit to one half of a dist. 
    Inps : tdata - array, tbins - array with reasonable defs (use the bin defs
        used to visualize the data in the first place)
    direction : which side of the distribution to fit the Gaussian to
    doplot : do you need to see the plots?
    Returns two arrays with mus and sigmas of the form 
    [lower bound, middle value, upper bound]
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from scipy.optimize import curve_fit
    
    # get a rough estimate of the peak
    pops, _ = np.histogram(tdata,tbins)
    bins = get_binmids(tbins)
    if peak: tpeak=peak
    else: tpeak = bins[np.argmax(pops)]  
    if edgetouse == 'upper':
        tpeak = tbins[(tpeak < tbins)][0]
    else:
        tpeak = tbins[(tpeak > tbins)][-1]
    # print('Approx peak = ',tpeak)
       
    # Which half of the histogram to use?
    if direction == 'left':
        tdist_1 = tdata[np.where(tdata < tpeak)[0]]
        tdist_2 = tpeak + (tpeak - tdist_1)
    else:
        tdist_1 = tdata[np.where(tdata > tpeak)[0]]
        tdist_2 = tpeak - (tdist_1 - tpeak)

    tdist = np.concatenate([tdist_1, tdist_2])
    mu0, sig0 = norm.fit(tdist)
            
    # fit a gaussian
    pops_half, _ = np.histogram(tdist,tbins)
    amp0 = np.nanmax(pops_half)
    popt, pcov = curve_fit(gaussian, bins, pops_half, p0=[amp0,mu0,sig0])
    # Extract the parameters
    amplitude, mean, stddev = popt
       
    if doplot:
        plt.figure(figsize=(6,4),layout='compressed')
        plt.hist(tdata, bins=tbins, density=False, alpha=0.6, color='y', label='actual data')
        plt.hist(tdist, bins=tbins, density=False, alpha=0.6, color='g', label='half histogram')
        x = np.linspace(tbins[0], tbins[-1], len(tbins))
        plt.plot(x, gaussian(x, *popt), 'r-', label='Fitted Gaussian')
        plt.axvline(tpeak, c='r', ls='-', lw=1, label="Peak est. = {0:.4f}".format(tpeak))
        plt.axvline(mean, c='k', ls='--', lw=1, label="mu = {0:.4f}".format(mean))
        plt.axvline(mean+stddev, c='k', ls='-.', lw=1, label="1sigma = {0:.4f}".format(stddev))
#         plt.xlabel('Value')
#         plt.ylabel('N')
        plt.legend(fontsize=12)
        plt.show()
           
    print(f"Fitted Gaussian parameters: amplitude={amplitude}, mean={mean}, stddev={stddev}")               

    return np.array([amplitude, mean, abs(stddev)]), tdist


def hist_peaks(tdata,tbins):
    '''
    Returns the peak, upper and lower fwhm locations
    of a histogram
    input : data (array) and binning scheme (arr)
    return : peak, [low_fwhm, up_fwhm], [sigdn,sigup]
    '''
    pops, bins = np.histogram(tdata, bins=tbins)
    peak = bins[np.argmax(pops)]
    pkerr = np.array([bins[np.argmax(pops)+1] - peak, peak - bins[np.argmax(pops)-1]])

    for ff in range(0,np.argmax(pops)):
        if pops[ff]> np.nanmax(pops)/2 : break
    if np.argmax(pops) == 0: ff = 0
    low_fwhm = (bins[ff]+bins[ff-1])/2
    
    # upper half
    for ff in range(np.argmax(pops), len(pops)):
        if pops[ff]< np.nanmax(pops)/2 : break
    up_fwhm = (bins[ff]+bins[ff-1])/2
    
    sigup = (up_fwhm - peak )/1.1775
    sigdn = (peak - low_fwhm)/1.1775

    print('Peak : {0:.4e} +/- ({1:.4e},{2:.4e})'.format(peak,pkerr[0],pkerr[1]))
    print('FWHM (lower) : {0:.4e}, FWHM (upper) : {1:.4e}'.format(low_fwhm, up_fwhm))
    print('Sigma (lower) : {0:.4e}, Sigma (upper) : {1:.4e}'.format(sigdn,sigup))

    
    return peak, pkerr, [low_fwhm, up_fwhm], [sigdn,sigup]


def element_counter(arr,elist=None,density=False):
    '''
    Returns a dictionary of instances of input elements
    in an array. If the list of elements is None or empty,
    returns count of unique elements in the array.
    
    element_counter(arr,density=False,elist=None)
    return {element:count,...}
    '''
    import numpy as np
    arr = np.array(arr)
    if elist is None or len(elist) == 0:
        elist = np.unique(arr)
        
    res = {}
    
    if density:
        for ee in elist: 
            res[ee] = len(np.where( arr == ee )[0])/len(arr)
    else:
        for ee in elist: 
            res[ee] = len(np.where( arr == ee )[0])
                      
    return res



def ridgelines(xdata,ydata,xbins,ybins):
    '''
    Calculate the peak of a 2d distribution
    returns ridge_mid, ridge_up, ridge_dn, binpops
    '''

    xbinmids, xbinwidth = get_binmids(xbins, True)
    
    ridge_dn, ridge_mid, ridge_up = \
                np.zeros(len(xbinmids)), np.zeros(len(xbinmids)), np.zeros(len(xbinmids))
    binpops = np.zeros(len(xbinmids))
    poplim = 0                        

    # binning

    # first bin
    ii = 0
    roi = (xdata < xbins[0])
    tkeys = np.where(roi==True)[0]

    tcnt, tbins = np.histogram(ydata[tkeys],bins=ybins)

    # ridgelines
    ridge_mid[0] = tbins[np.argmax(tcnt)]
    # lower half
    for ff in range(0,np.argmax(tcnt)):
        if tcnt[ff]> np.nanmax(tcnt)/2 : break
    if np.argmax(tcnt) == 0: ff = 0
    ridge_dn[0] = tbins[ff]
    # upper half
    for ff in range(np.argmax(tcnt), len(tcnt)):
        if tcnt[ff]< np.nanmax(tcnt)/2 : break
    ridge_up[0] = tbins[ff]

    binpops[0] = len(tkeys)


    for ii in range(len(xbins)-1):
        sup, sdn = xbins[ii+1], xbins[ii]
        roi = (sdn < xdata) & (xdata < sup)
        tkeys = np.where(roi==True)[0]
        binpops[ii+1] = len(tkeys)

        tcnt, tbins = np.histogram(ydata[tkeys],bins=ybins)

        # ridgelines
        ridge_mid[ii+1] = tbins[np.argmax(tcnt)]
        # lower half
        for ff in range(0,np.argmax(tcnt)):
            if tcnt[ff]> np.nanmax(tcnt)/2 : break
        if np.argmax(tcnt) == 0: ff = 0
        ridge_dn[ii+1] = tbins[ff]
        # upper half
        for ff in range(np.argmax(tcnt), len(tcnt)):
            if tcnt[ff]< np.nanmax(tcnt)/2 : break
        ridge_up[ii+1] = tbins[ff]
        
        


    # last bin
    ii = len(xbins)
    roi = (xdata > xbins[-1])
    tkeys = np.where(roi==True)[0]

    tcnt, tbins = np.histogram(ydata[tkeys],bins=ybins)
    # ridgelines
    ridge_mid[-1] = tbins[np.argmax(tcnt)]
    # lower half
    for ff in range(0,np.argmax(tcnt)):
        if tcnt[ff]> np.nanmax(tcnt)/2 : break
    if np.argmax(tcnt) == 0: ff = 0
    ridge_dn[-1] = tbins[ff]
    # upper half
    for ff in range(np.argmax(tcnt), len(tcnt)):
        if tcnt[ff]< np.nanmax(tcnt)/2 : break
    ridge_up[-1] = tbins[ff]

    binpops[-1] = len(tkeys)
    
    return ridge_mid, ridge_up, ridge_dn, binpops



def sample_werrors(X,N=100):
    '''
    sample_werrors(X,N=100): return xN 
    Generate N samplings from the skewed Gaussian centered at x
    and deviations and xerr_up and xerr_dn.
    Default N = 100

    '''
    # First generate 100 samplings for each source
    # y (sfr or mstar) will have a skewed normal dist
    if np.shape(X)[0] == 1:  m,merrup,merrdn = X, 0.1*X, 0.1*X
    elif np.shape(X)[0] == 2:  
        m,merrup = X
        merrdn = merrup
    elif np.shape(X)[0] == 3:  m,merrup,merrdn = X
    else: 
        print('Too many inputs in X')
        return None, None

    tmN = np.zeros(N*len(m))
    for ii in range(len(m)):
        # Get 4x samples in both positive and negatives
        # Select 50 from each
        tm = np.zeros(N)
        tm_upi = np.random.normal(m[ii], merrup[ii], 2*N)
        tm_upi_N_2 = tm_upi[np.where(tm_upi >= m[ii])[0]][:50]

        tm_dni = np.random.normal(m[ii], merrdn[ii], 2*N)
        tm_dni_N_2 = tm_dni[np.where(tm_dni <= m[ii])[0]][:50]

        for tt in range(N//2):
            tm[tt] = tm_upi_N_2[tt]
            tm[tt+1] = tm_dni_N_2[tt]

        tmN[ii*N : (ii+1)*N] = tm
            
    return tmN    



def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    import matplotlib.colors as colors
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp    


# main sequence parametrization
def main_seq(M,z = 0):
    r = np.log10(1+z)
    m = M - 9

    m0 = 0.5 #+/- 0.07
    m1 = 0.36 #+/- 0.3

    a0 = 1.5 #+/- 0.15
    a1 = 0.3 #+/- 0.08
    a2 = 2.5 #+/- 0.6

    logSFR_ms = m - m0 + a0*r - a1*(np.maximum(0, (m-m1-a2*r) )**2)

    return logSFR_ms    




def norm_hist(tdat, bins, **kwargs):
    # plots a histogram with peak = 1 
    # standard plt.hist inputs should work
    # Note: Any input weight will be overwritten
    # range will not work
    import numpy as np
    import matplotlib.pyplot as plt
    
    yhist, xhist = np.histogram(tdat, bins=bins);
    tdat, tbins, tpatches = plt.hist(tdat, bins=bins, weights=[1/yhist.max()]*len(tdat), 
                                     rasterized=True, **kwargs);
    
    return tdat, tbins, tpatches    

def norm_weights(x,bins):
    # returns a len(x) sized array containing normalizing factors
    import numpy as np
    yhist, xhist = np.histogram(x, bins=bins);
    return [1/yhist.max()]*len(x)    


def skewed_dist(s, seup, sedn, N = 100):
    '''
    Get N realizations of s sampled from a skewed distribution
    '''
    n = len(s)
    
    rands_s = np.random.randn(N, n)
    svals = np.zeros((N, n))
    svals_eup = np.zeros((N, n))
    svals_edn = np.zeros((N, n))
    
    for i in range(N):
        svals[i,:] = s
        svals_eup[i,:] = seup
        svals_edn[i,:] = sedn
        
    poss = (rands_s > 0)
    negs = (rands_s < 0)
    svals[poss] = svals[poss] + (svals_eup[poss]*rands_s[poss])
    svals[negs] = svals[negs] + (svals_edn[negs]*rands_s[negs])

    return svals.flatten()    



def trunc(values, decs=0):
    '''
    Given an array of floats, truncates it to the desired
    number of decimal places
    '''
    return np.trunc(values*10**decs)/(10**decs)    


def reshape_array(original_array,target_shape):
    import numpy as np
    from scipy.interpolate import griddata
    # Create the target grid
    x, y = np.meshgrid(np.linspace(0, 1, target_shape[1]), np.linspace(0, 1, target_shape[0]))

    # Reshape the original array to 1D
    x_orig, y_orig = np.meshgrid(np.linspace(0, 1, original_array.shape[1]), 
                                 np.linspace(0, 1, original_array.shape[0]))
    points = np.column_stack((x_orig.ravel(), y_orig.ravel()))
    values = original_array.ravel()

    # Perform bilinear interpolation
    return np.array(griddata(points, values, (x, y), method='linear'))



def flux_nu_to_Lnu_Lsol(rest_wave, phot_flux, zred):
    import numpy as np
    # Astropy to calculate cosmological params
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    # call the cosmo object from astropy
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    # Calculate the redshift comoving distance
    lumdist  = cosmo.luminosity_distance(zred).value     # in mpc
    lumdist_cm = lumdist*3.086e24 # in cm

    # Convert to rest frame
    phot_flux_rest_maggies = np.array(phot_flux)/(1+zred) # in maggies

    Lo = 3.826e33
    phot_Lnu_Lsol = ((4*np.pi*3631.e-23)/Lo)*phot_flux_rest_maggies*lumdist_cm**2

    return phot_Lnu_Lsol # in Lsol/Hz

def Lnu_Lsol_to_Llambda_Lsol(rest_wave_A, Lnu_Lsol):
    import numpy as np
    c  = 299792458
    Llambda_Lsol = (1.e10 *Lnu_Lsol*c)/( rest_wave_A**2)
    return Llambda_Lsol

def get_ptiles(arr):
    '''
    return basic stats
    get_ptiles(arr)
    
    return np.array([p16,p50,p84]), np.array([sig,sigup, sigdn])
    '''
    import numpy as np
    p16,p50,p84 = np.percentile(arr,[16,50,84])
    sigup, sigdn = p84 - p50, p50 - p16
    sig = (sigup+sigdn)/2
    
    return np.array([p16,p50,p84]), np.array([sig,sigup, sigdn])    

def print_ptiles(arr, dtype='scien', header=True, tag = None):
    '''
    print and return basic stats
    print_ptiles(arr, dtype='scien(DEFAULT) | int | float', header=True, tag = None)
    
    return np.array([p16,p50,p84]), np.array([sig,sigup, sigdn])
    '''
    import numpy as np
    p16,p50,p84 = np.percentile(arr,[16,50,84])
    sigup, sigdn = p84 - p50, p50 - p16
    sig = (sigup+sigdn)/2

    if tag is not None:
        tag = (tag[:14] + '..') if len(tag) > 16 else tag
        ptag = '{0:17} ||'.format(tag)
        htag = '{0:17} ||'.format('')
        nspaces = 110
    else:
        ptag, htag, nspaces = '', '', 96
    
    if header:
        print(htag+'{0:^10} | {1:^10} | {2:^10} | {3:^10} | {4:^10} | {5:^10} | {6:^10}'
              .format('p16','p50','p84','sig','3sig','sigup', 'sigdn'))
        print("-"*nspaces)
        
    if dtype == 'int':
        print(ptag+'{0:^10} | {1:^10} | {2:^10} | {3:^10} | {4:^10} | {5:^10} | {6:^10}'
              .format(int(p16),int(p50),int(p84),int(sig),int(sigup), int(sigdn)))
    elif dtype == 'float':
        print(ptag+'{0:^10.3f} | {1:^10.3f} | {2:^10.3f} | {3:^10.3f} | {4:^10.3f} | {5:^10.3f} | {6:^10.3f}'
              .format(p16,p50,p84,sig,3*sig,sigup, sigdn))
    elif dtype == 'scien':
        print(ptag+'{0:^10.3e} | {1:^10.3e} | {2:^10.3e} | {3:^10.3e} | {4:^10.3e} | {5:^10.3e} | {5:^10.3e}'
              .format(p16,p50,p84,sig,3*sig,sigup, sigdn))
    else:
        print('invalid stype for print, defaulting to scien')
        print(ptag+'{0:^10.3e} | {1:^10.3e} | {2:^10.3e} | {3:^10.3e} | {4:^10.3e} | {5:^10.3e} | {5:^10.3e}'
              .format(round(p16,4),p50,p84,sig,3*sig,sigup, sigdn))
    
    return np.array([p16,p50,p84]), np.array([sig,sigup, sigdn])

def moving_avg(arr,N):
    import numpy as np
    return np.convolve(arr, np.ones(N)/N, mode='valid')    


def arr_check(var):
    typevar = type_str(var)
    if 'list' in typevar or 'array' in typevar : return np.array(var)
    elif 'float' in typevar or 'int' in typevar or 'bool' in typevar: return np.array([var])
    else: 
        raise ValueError("Input(s) must be int/float/array.")
        return np.nan



def getcmap(dataarr, clmap='Spectral_r', nbins=10):
    import numpy as np
    import matplotlib.pyplot as plt
    
    clarr = eval("plt.cm."+clmap+"(np.linspace(0,1,"+str(nbins)+"))")
    clpt = plt.scatter(np.ones(len(dataarr)), np.ones(len(dataarr)), 
               c = dataarr, cmap=clmap)
    clbins = np.linspace(np.nanmin(dataarr),np.nanmax(dataarr),nbins+1)
    
    plt.clf()
    plt.close()
    
    return clarr, clpt, clbins

def getcmapid(x, clbins):
    ncmap = len(clbins) - 1
    if not np.isfinite(x): return np.nan
    elif x < clbins[1]: return 0
    elif x > clbins[-2]: return ncmap - 1
    else: 
        for ii in range(ncmap):
            if (clbins[ii] < x) & (x < clbins[ii+1]): return ii