import numpy as np
import emcee
import corner
from scipy.optimize import curve_fit

def model(X, logLc,B,Y): 
    logsfr, logmass10 = X
    return logLc + B*logsfr + Y*logmass10

def log_prior(theta):
    logLc,B,Y = theta
    if 15 < logLc and logLc < 25 and -5 < B and B < 5 and -5 < Y and Y < 5: return 0
    else: return -np.inf

from scipy.stats import norm
def log_likelihood(theta, X, F):
    logLc,B,Y = theta
    logL150,logL150err = F
    m = model(X, logLc,B,Y)
    return np.sum(norm.logpdf(logL150, loc=m, scale=logL150err))

def log_probability(theta, X, F):
    logP = log_prior(theta)
    if not np.isfinite(logP):
        return -np.inf
    return logP + log_likelihood(theta, X, F)

def run_emcee(X,F,init_guess,changeinit=True, nchains= 32, niters=3000, nburnin=500,plots=True):
    
    # Get initial guess
    m_sfr_arr,m_m_arr = X
    logL150,logL150err = F
    
    if changeinit:
        logLc_init, B_init, Y_init = curve_fit(model,X,logL150,init_guess,logL150err)[0]
    else:
        logLc_init, B_init, Y_init = init_guess
        
    print('Initial guesses : log L_c = {0:.3f}, beta = {1:.3f},  gamma = {2:.3f}' \
          .format(logLc_init, B_init, Y_init))
    
    posn_logLc = logLc_init + 0.1*logLc_init*(np.random.rand(nchains)-0.5)
    posn_B = B_init + 0.1*B_init*(np.random.rand(nchains)-0.5)
    posn_Y = Y_init + 0.1*Y_init*(np.random.rand(nchains)-0.5)
    
    # Turn into a single array
    positions = np.array([posn_logLc,posn_B,posn_Y]).T
    nwalkers, ndim = np.shape(positions)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(X,F)
    )
    
    state = sampler.run_mcmc(positions, nburnin, progress=True)
    sampler.reset()
    result = sampler.run_mcmc(state, niters, progress=True)
    
    res_logLc = sampler.flatchain[:,0]
    res_B = sampler.flatchain[:,1]
    res_Y = sampler.flatchain[:,2]

    weights = sampler.flatlnprobability
    if plots:
        fig, ax = plt.subplots(3,1,figsize=(11,8),sharex=True)
        ax[0].plot(res_logLc,'.k',ms=0.5)
        ax[0].set_ylabel(r'$\log L_c$')
        ax[1].plot(res_B,'.k',ms=0.5)
        ax[1].set_ylabel(r'$\beta$')
        ax[2].plot(res_Y,'.k',ms=0.5)
        ax[2].set_ylabel(r'$\gamma$')
        plt.tight_layout()
        plt.show()
    
    print('Final estimate : log L_c = {0:.3f}, beta = {1:.3f},  gamma = {2:.3f}' \
          .format(weighted_quantile(res_logLc,[0.50],weights)[0],
        weighted_quantile(res_B,[0.50],weights)[0],
        weighted_quantile(res_Y,[0.50],weights)[0]))
    
    if plots:
        rmp = corner.corner(sampler.flatchain, labels=[r'$\log L_c$',r'$\beta$',r'$\gamma$'])
#                     truths=[res_logLc,res_B,res_Y])
    
    return res_logLc,res_B,res_Y,weights        