import os, sys
from tqdm.notebook import tqdm_notebook
import fitsio
from fitsio import FITS,FITSHDR
from astropy.io import fits
import numpy as np

def loadfits2dict(fitsfile,colname='sour_id'):
    fitshdu = fits.open(fitsfile)
    fitsarr = fitsio.read(fitsfile)
    fitscols = fitshdu[1].columns
    fitskeys = {fitscols[ii].name:ii for ii in range(len(fitscols)) }
    fitsdict = {}
    for ths in tqdm_notebook(fitsarr): fitsdict[ths[fitskeys[colname]]] = ths
    for aa in list(fitsdict): 
        if aa < 0: del fitsdict[aa]
    return fitsdict, fitskeys


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

def half_hist_to_gauss(tdata,tpeak,twhich='left',doplot=True):
    '''
    To return mean and std of a 1-d normal distribution with upper
    and lower bounds fit to one half of a dist. 
    Inps : tdata - array, tbins - array with reasonable defs (use the bin defs
        used to visualize the data in the first place)
    twhich : which side of the distribution to fit the Gaussian to
    doplot : do you need to see the plots?
    Returns two arrays with mus and sigmas of the form 
    [lower bound, middle value, upper bound]
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    
    # get a rough estimate of the peak
    if twhich == 'left':
        tdist_1 = tdata[np.where(tdata < tpeak)[0]]
        tdist_2 = tpeak + (tpeak - tdist_1)
    else:
        tdist_1 = tdata[np.where(tdata > tt)[0]]
        tdist_2 = tpeak - (tdist_1 - tpeak)

    tdist = np.concatenate([tdist_1, tdist_2])
            
    (mu, sigma) = norm.fit(tdist)
    tptil16, tptil50, tptil84 = np.nanpercentile(tdist, [16, 50, 84])

        
    N = int(np.sqrt(np.sum((mu-3*sigma < tdist) & (tdist < mu+3*sigma) )))
    nbins = np.linspace(mu-5*sigma,mu+5*sigma,N)
        
    if doplot:
        plt.figure()
        fig, axes = plt.subplots(1,3,figsize=(12,4), sharey=True)
        
        tplt = axes[0]
        tplt.hist(tdist, bins=nbins, color='skyblue',density=True);
        tplt.hist(tdata, bins=nbins, color='orange',density=True, alpha=0.4);
        tplt.set_ylabel('propto N')
        tplt.set_xlabel('x bins')
        tplt.set_title('Original dist')

        tplt = axes[1]
        tplt.hist(tdist, bins=nbins, color='skyblue',density=True);
        tplt.axvline(mu, c='firebrick', ls='--',label='peak')        
        tplt.axvline(mu-sigma, c='firebrick', ls='-.')        
        tplt.axvline(mu+sigma, c='firebrick', ls='-.', label='+/- sigma')        
        tplt.plot(nbins, norm.pdf(nbins,mu, sigma),lw=1,c='purple')
        
        tplt.set_xlabel('x bins')
        tplt.set_title('Fitting Gaussian')
        
        tplt = axes[2]
        tplt.hist(tdist, bins=nbins, color='skyblue',density=True);
        tplt.axvline(tptil16, c='firebrick', ls='--')        
        tplt.axvline(tptil50, c='firebrick', ls='--')        
        tplt.axvline(tptil84, c='firebrick', ls='--')   
        tplt.set_xlabel('x bins')
        tplt.set_title('Percentiles')
        
        plt.tight_layout()
                
    print('mu = {0:.3f}, sigma = {1:.3f}'.format(mu, sigma))        
    print('16ptile = {0:.3f}, 50 = {1:.3f}, 84 = {2:.3f}'.format(tptil16, tptil50, tptil84))                  

    return np.array([mu, sigma]),np.array([tptil16, tptil50, tptil84])


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


def element_counter(arr,elist=None):
    '''
    Returns a dictionary of instances of input elements
    in an array. If the list of elements is None or empty,
    returns count of unique elements in the array.
    
    element_counter(arr,elist=None)
    return {element:count,...}
    '''
    import numpy as np
    arr = np.array(arr)
    if elist is None or len(elist) == 0:
        elist = np.unique(arr)
        
    res = {}
    
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




def norm_hist(x, bins=10, density=False, weights=None, cumulative=False, rasterized=False,
              bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, 
              log=False, color=None, label=None, stacked=False, *, data=None, **kwargs):
    # plots a histogram with peak = 1 
    # standard plt.hist inputs should work
    # Note: Any input weight will be overwritten
    # range will not work
    import numpy as np
    import matplotlib.pyplot as plt
    
    yhist, xhist = np.histogram(x, bins=bins);
    tdat, tbins, tpatches = plt.hist(x, bins=bins, density=density, cumulative=cumulative, 
              bottom=bottom, histtype=histtype, align=align, orientation=orientation, rwidth=rwidth, 
              log=log, color=color, label=label, stacked=stacked,
              weights=[1/yhist.max()]*len(x),rasterized=rasterized);
    
    return tdat, tbins, tpatches    

def norm_weights(x,bins):
    # returns a len(x) sized array containing normalizing factors
    import numpy as np
    yhist, xhist = np.histogram(x, bins=bins);
    return [1/yhist.max()]*len(x)    