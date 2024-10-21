import numpy as np
from scipy.optimize import minimize
import emcee
import corner
import matplotlib.pyplot as plt

def log_likelihood(theta, x, y, yerr):
    c = theta
    model = x + c
    sigma2 = yerr**2 + model**2 # * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 )

def log_prior(theta):
    c = theta
    if -5 < c < 5: return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)        

def max_likelihood(x, y, yerr):
    np.random.seed(42)
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([0]) + 0.1 * np.random.randn(1)
    soln = minimize(nll, initial, args=(x, y, yerr))
    c_ml = soln.x

    print("Maximum likelihood estimates: c = {0:.3f}".format(c_ml[0]))
    return c_ml[0]

def run_emcee(x, y, yerr, nburnin=2000, niter=10000):
    '''Return median likelihood, sigma up, sigma down'''
    # init guess
    init_guess = max_likelihood(x, y, yerr)
    pos = init_guess + 1e-2 * np.random.randn(32, 1)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(x, y, yerr)
    )

    state = sampler.run_mcmc(pos, nburnin, progress=True)
    sampler.reset()
    result = sampler.run_mcmc(state, niter, progress=True)

    fig, axes = plt.subplots(ndim,1, figsize=(8, 3), sharex=True)
    samples = sampler.get_chain()
    labels = ["c"]
    for i in range(ndim):
        ax = axes
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes.set_xlabel("step number");

    tau = sampler.get_autocorr_time()
    print('Autocorr time:',tau)

    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
    print(flat_samples.shape)

    fig = corner.corner(flat_samples, labels=labels, truths=[0]);

    from IPython.display import display, Math

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))

    return mcmc[1], q[1], q[0]

