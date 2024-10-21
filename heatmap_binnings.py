import numpy as np
# --------------------------------------------------------- #
#   Binmids
# --------------------------------------------------------- #
def get_binmids(tarr,borders=False):
    tbinmids = tarr + (tarr[1] - tarr[0])/2
    if borders: res_arr = np.append([tarr[0] - (tarr[1] - tarr[0])/2],tbinmids)
    else: res_arr = tbinmids[:-1] 
    
    return res_arr, np.ones(len(res_arr)) * ((tarr[1] - tarr[0])/2)

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

# --------------------------------------------------------- #
#   Heatmap Single x bin
# --------------------------------------------------------- #
def heatmap_singlebin(y,yerrup,yerrdn,z,zerr,bins,lowstatlim=100):

    # First generate 100 samplings for each source
    # y (sfr or mstar) will have a skewed normal dist
    N = 100
    
    # tyN and tzN are the arrays that will go to make the heatmap
    tyN = np.zeros(N*len(y))
    tzN = np.zeros(N*len(z))
    
    for ii in range(len(y)):
        # Get 4x samples in both positive and negatives
        # Select 50 from each
        ty = np.zeros(N)
        ty_upi = np.random.normal(y[ii], yerrup[ii], 2*N)
        ty_upi_N_2 = ty_upi[np.where(ty_upi >= y[ii])[0]][:50]

        ty_dni = np.random.normal(y[ii], yerrdn[ii], 2*N)
        ty_dni_N_2 = ty_dni[np.where(ty_dni <= y[ii])[0]][:50]


        for tt in range(N//2):
            ty[tt] = ty_upi_N_2[tt]
            ty[tt+1] = ty_dni_N_2[tt]

        tyN[ii*N : (ii+1)*N] = ty

    for ii in range(len(z)):
        tzN[ii*N : (ii+1)*N] = np.random.normal(z[ii], zerr[ii], N)
        
    # Once we have ty and tz ready, put the radio lum < 10^17 to =10^17.01
    tzN[np.where(tzN < 1.e17)[0]] = 10**(17.01)
    
    # Make the heatmap
    pop_matrix, ybins, zbins = np.histogram2d(np.log10(tyN),
        np.log10(tzN),bins=(bins['ybins'], bins['zbins']))
    
    # Get binmids for calculating the quantiles
    ybinmids, ybinerrs = get_binmids(bins['ybins'])
    zbinmids, zbinerrs = get_binmids(bins['zbins'])

    # arrays for the percentiles
    z_p16, z_p50, z_p84 = np.zeros(len(ybinmids)), np.zeros(len(ybinmids)), np.zeros(len(ybinmids))
    for ii in range(len(ybinmids)):
        # which radio luminosity bins have objects less than 15 (x100)
        # and no of objs is no inf or nan
        good_indices = np.where(np.isfinite(pop_matrix[ii]))[0]
        if(np.nansum(pop_matrix[ii]) > lowstatlim and len(good_indices)>0):
            z_p16[ii], z_p50[ii], z_p84[ii] = weighted_quantile(zbinmids[good_indices],
                                    [.16,.50,.84],pop_matrix[ii][good_indices])
        else : z_p16[ii], z_p50[ii], z_p84[ii] = np.nan, np.nan, np.nan
            
    res = {'heatmap': pop_matrix, 'y' : tyN, 'z' : tzN,
        'p16': z_p16, 'p50': z_p50, 'p84': z_p84,}
            
    return res

# --------------------------------------------------------- #
# Heatmap Multiple x binnings
# --------------------------------------------------------- #
def full_bins(inpdata,bins,low_stat_lims = [100,100]):
    x = inpdata['x']
    y, yerrup, yerrdn = inpdata['y'], inpdata['yerrup'], inpdata['yerrdn']
    z, zerr = inpdata['z'], inpdata['zerr']
    
    xbins, ybins, zbins = bins['xbins'], bins['ybins'], bins['zbins']
    
    x_digitized = np.digitize(np.log10(x), xbins)
    
    # Get binmids for calculating the quantiles
    xbinmids, xbinerrs = get_binmids(xbins)
    ybinmids, ybinerrs = get_binmids(ybins)
    zbinmids, zbinerrs = get_binmids(ybins)
    
    # empty dict to stash quantiles of each x bin  
    binned_ptiles = {}
      
    # final loop
    for ii in tqdm.notebook.tqdm(range(len(xbins))):
        inds = np.where(x_digitized == ii)[0]
        
        if (len(inds) > low_stat_lims[0]):
        
            yi, yerrupi, yerrdni = y[inds], yerrup[inds], yerrdn[inds] 
            zi, zerri = z[inds], zerr[inds]

            resi = heatmap_singlebin(y=yi, yerrup=yerrupi, yerrdn=yerrdni,
                        z=zi, zerr=zerri, bins=bins,lowstatlim = low_stat_lims[1])

            binned_ptiles[ii] = {16: resi['p16'], 50: resi['p50'], 84:resi['p84']}
        
    return binned_ptiles