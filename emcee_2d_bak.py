import numpy as np
def model(X, m, c): 
    return c + m*X

def log_prior(theta, lims):
    m, c = theta
    mdn, mup, cdn, cup = lims
    if mdn < m and m < mup and cdn < c and c < cup: return 0
    else: return -np.inf
    
def log_likelihood(theta, X, F):
    m, c = theta
    x, xerr = X  # dependant variable
    y, yedn, yeup = F # observed quantity
    
    m = model(x, m ,c)
    
    chi2arr = np.zeros(len(y))
    # m is analogous to x
    for ii in range(len(y)):
        # if observed value is greater than the model value
        # use the upper error bound, else use lower
        chi2arr[ii] = 0.5*(y[ii] - m[ii])**2/(xerr[ii]**2+yeup[ii]**2) + \
                      0.5*(y[ii] - m[ii])**2/(xerr[ii]**2+yedn[ii]**2)
    chi2 = np.sum(chi2arr)
            
    return np.log10(1/chi2)

def log_probability(theta, lims, X, F):
    logP = log_prior(theta, lims)
    if not np.isfinite(logP):
        return -np.inf
    return logP + log_likelihood(theta, X, F)


def run_emcee(X,F,init_guess, lims, nchains= 32, niters=3000, nburnin=500):
    '''
    Inputs
    X -> x or (x, xerr). If xerr is not specified, xerr is assumed to be 10% of x
    Y -> y [Just the observation] -> yerr is assumed to be 10% of y in this case OR
        (y, yerr) -> yeup, yedn = yerr in this case
        (y, yedn, yeup)
    lims -> valid limits on the variables being fit for
        2*N dim array, N being number of variables
        form : [v1dn, v1up, v2dn, v2up,...]
    '''
    from scipy.optimize import curve_fit
    import emcee
    import matplotlib.pyplot as plt

    # Independant variable
    if np.shape(X)[0] == 1: x, xerr = X, 0.1*X
    else: x, xerr = X
    X = (x, xerr)

    # observation
    if np.shape(F)[0] == 1:
        y = F
        yeup, yedn = 0.1*y, 0.1*y
        
    elif np.shape(F)[0] == 2:
        y, yeup = F
        yedn = yeup
        
    elif np.shape(F)[0] == 3:
        y, yedn, yeup = F

    F =  (y, yedn, yeup)

    # Get initial guess
    init_guess = curve_fit(model,x,y,init_guess,np.sqrt((yeup**2+yedn**2)/2))
    posn_m = init_guess[0][0] + 0.05*init_guess[0][0]*(np.random.rand(nchains)-0.5)
    posn_c = init_guess[0][1] + 0.05*init_guess[0][1]*(np.random.rand(nchains)-0.5)
    
    # Turn into a single array
    positions = np.array([posn_m,posn_c]).T
    nwalkers, ndim = np.shape(positions)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(lims, X,F)
    )
    
    state = sampler.run_mcmc(positions, nburnin, progress=True)
    sampler.reset()
    result = sampler.run_mcmc(state, niters, progress=True)
    
    res_m, res_c = sampler.flatchain[:,0], sampler.flatchain[:,1]

    weights = 10**sampler.flatlnprobability
    
    # plot the chains to see good fit
    fig, ax  =plt.subplots(1,2, figsize=(10,4))
    tcnt, tbins, textra = ax[0].hist(res_m, bins=50)
    peak_m = tbins[np.argmax(tcnt)]
    ax[0].axvline(weighted_quantile(res_m,[0.50],weights)[0], c='k')
    ax[0].axvline(peak_m, c='r')
    ax[0].set_xlabel('m')

    tcnt, tbins, textra = ax[1].hist(res_c, bins=50)
    peak_c = tbins[np.argmax(tcnt)]
    ax[1].axvline(weighted_quantile(res_c,[0.50],weights)[0], c='k')
    ax[1].axvline(peak_c, c='r')
    ax[1].set_xlabel('c')
    
    # final values
    print('Final estimate : m = {0:.3f}, c = {1:.3f}' 
      .format(weighted_quantile(res_m,[0.50],weights)[0],weighted_quantile(res_c,[0.50],weights)[0]))
    
    return res_m,res_c,[peak_m,peak_c],weights


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