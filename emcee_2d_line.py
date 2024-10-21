import numpy as np
def model(X, m, c): 
    return c + m*X

def log_prior(theta):
    m, c = theta
    if 0.5 < m and m < 1.5 and -3 < c and c < 3: return 0
    else: return -np.inf
    
def log_likelihood(theta, X, F):
    m, c = theta
    x, xerr = X  # dependant variable
    y, yerr = F  # observations
    m = model(x, m ,c)
    if xerr is None: xerr = 0.1*x
    if yerr is None: yerr = 0.1*y
    chi2 = np.sum((y - m)**2/(xerr**2+yerr**2))
    return np.log10(1/chi2)

def log_probability(theta, X, F):
    logP = log_prior(theta)
    if not np.isfinite(logP):
        return -np.inf
    return logP + log_likelihood(theta, X, F)


def run_emcee(X,F,init_guess, nchains= 32, niters=3000, nburnin=500):
    from scipy.optimize import curve_fit
    import emcee
    import matplotlib.pyplot as plt

    x,xerr = X  # inependent variable
    y,yerr = F  # observation

    # Get initial guess
    init_guess = curve_fit(model,x,y,init_guess,yerr)
    posn_m = init_guess[0][0] + 0.1*init_guess[0][0]*(np.random.rand(nchains)-0.5)
    posn_c = init_guess[0][1] + 0.1*init_guess[0][1]*(np.random.rand(nchains)-0.5)
    
    # Turn into a single array
    positions = np.array([posn_m,posn_c]).T
    nwalkers, ndim = np.shape(positions)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(X,F)
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