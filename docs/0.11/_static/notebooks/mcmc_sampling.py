
# coding: utf-8

# # Fitting and error estimation with MCMC
# 
# ## Introduction
# 
# The goal of Markov Chain Monte Carlo (MCMC) algorithms is to approximate the posterior distribution of your model parameters by random sampling in a probabilistic space. For most readers this sentence was probably not very helpful so here we'll start straight with and example but you should read the more detailed mathematical approaches of the method [here](https://www.pas.rochester.edu/~sybenzvi/courses/phy403/2015s/p403_17_mcmc.pdf) and [here](https://github.com/jakevdp/BayesianAstronomy/blob/master/03-Bayesian-Modeling-With-MCMC.ipynb).
# 
# ### How does it work ?
# 
# The idea is that we use a number of walkers that will sample the posterior distribution (i.e. sample the Likelihood profile).
# 
# The goal is to produce a "chain", i.e. a list of $\theta$ values, where each $\theta$ is a vector of parameters for your model.<br>
# If you start far away from the truth value, the chain will take some time to converge until it reaches a stationary state. Once it has reached this stage, each successive elements of the chain are samples of the target posterior distribution.<br>
# This means that, once we have obtained the chain of samples, we have everything we need. We can compute the  distribution of each parameter by simply approximating it with the histogram of the samples projected into the parameter space. This will provide the errors and correlations between parameters.
# 
# 
# Now let's try to put a picture on the ideas described above. With this notebook, we have simulated and carried out a MCMC analysis for a source with the following parameters:<br>
# $Index=2.0$, $Norm=5\times10^{-12}$ cm$^{-2}$ s$^{-1}$ TeV$^{-1}$, $Lambda =(1/Ecut) = 0.02$ TeV$^{-1}$ (50 TeV) for 20 hours.
# 
# The results that you can get from a MCMC analysis will look like this :
# 
# <img src="images/gammapy_mcmc.png" width="800">
# 
# On the first two top panels, we show the pseudo-random walk of one walker from an offset starting value to see it evolve to a better solution.
# In the bottom right panel, we show the trace of each 16 walkers for 500 runs (the chain described previsouly). For the first 100 runs, the parameter evolve towards a solution (can be viewed as a fitting step). Then they explore the local minimum for 400 runs which will be used to estimate the parameters correlations and errors.
# The choice of the Nburn value (when walkers have reached a stationary stage) can be done by eye but you can also look at the autocorrelation time.
# 
# ### Why should I use it ?
# 
# When it comes to evaluate errors and investigate parameter correlation, one typically estimate the Likelihood in a gridded search (2D Likelihood profiles). Each point of the grid implies a new model fitting. If we use 10 steps for each parameters, we will need to carry out 100 fitting procedures. 
# 
# Now let's say that I have a model with $N$ parameters, we need to carry out that gridded analysis $N*(N-1)$ times. 
# So for 5 free parameters you need 20 gridded search, resulting in 2000 individual fit. 
# Clearly this strategy doesn't scale well to high-dimensional models.
# 
# Just for fun: if each fit procedure takes 10s, we're talking about 5h of computing time to estimate the correlation plots. 
# 
# There are many MCMC packages in the python ecosystem but here we will focus on [emcee](http://url), a lightweight Python package. A description is provided here : [Foreman-Mackey, Hogg, Lang & Goodman (2012)](https://arxiv.org/abs/1202.3665).

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.irf import load_cta_irfs
from gammapy.maps import WcsGeom, MapAxis
from gammapy.spectrum.models import ExponentialCutoffPowerLaw
from gammapy.image.models import SkyGaussian
from gammapy.cube.models import SkyModel
from gammapy.cube.simulate import simulate_dataset
from gammapy.utils.fitting import Fit

import emcee
import corner


# ## Simulate an observation
# 
# Here we will start by simulating an observation using the `simulate_dataset` method.

# In[ ]:


irfs = load_cta_irfs("$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits")


# In[ ]:


# Define sky model to simulate the data
spatial_model = SkyGaussian(lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg")

spectral_model = ExponentialCutoffPowerLaw(
    index=2,
    amplitude="3e-12 cm-2 s-1 TeV-1",
    reference="1 TeV",
    lambda_="0.05 TeV-1",
)

sky_model_simu = SkyModel(
    spatial_model=spatial_model, spectral_model=spectral_model
)
print(sky_model_simu)


# In[ ]:


# Define map geometry
axis = MapAxis.from_edges(
    np.logspace(-1, 2, 30), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(0, 0), binsz=0.05, width=(2, 2), coordsys="GAL", axes=[axis]
)

# Define some observation parameters
pointing = SkyCoord(0 * u.deg, 0 * u.deg, frame="galactic")


dataset = simulate_dataset(
    sky_model_simu, geom, pointing, irfs, livetime=20 * u.h, random_state=42
)


# In[ ]:


dataset.counts.sum_over_axes().plot(add_cbar=True);


# In[ ]:


# If you want to fit the data for comparison with MCMC later

# fit = Fit(dataset)
# result = fit.run(optimize_opts={"print_level": 1})


# ## Estimate parameter correlations with MCMC
# 
# Now let's analyse the simulated data.
# Here we just fit it again with the same model we had before as a starting point.
# The data that would be needed are the following: 
# - counts cube, psf cube, exposure cube and background model
# 
# Luckily all those maps are already in the Dataset object.
# 
# We will need to define a Likelihood function and define priors on parameters.<br>
# Here we will assume a uniform prior reading the min, max parameters from the sky model.

# In[ ]:


# Prior functions


def uniform_prior(value, umin, umax):
    """Uniform prior distribution."""
    if umin <= value <= umax:
        return 0.0
    else:
        return -np.inf


def normal_prior(value, mean, sigma):
    """Normal prior distribution."""
    return -0.5 * (2 * np.pi * sigma) - (value - mean) ** 2 / (2.0 * sigma)


# In[ ]:


def model_to_par(dataset):
    """
    Return a tuple of the factor parameters of all 
    free parameters in the dataset sky model.
    """
    pars = []
    for p in dataset.parameters.free_parameters:
        pars.append(p.factor)

    return pars


def par_to_model(dataset, pars):
    """Update model in dataset with a list of free parameters factors"""
    for i, p in enumerate(dataset.parameters.free_parameters):
        p.factor = pars[i]


def lnprior(dataset):
    """
    Return probability of parameter values according to prior knowledge.
    Parameter limits should be done here through uniform prior ditributions
    """
    logprob = 0
    for par in dataset.parameters.free_parameters:
        logprob += uniform_prior(par.value, par.min, par.max)

    return logprob


def lnprob(pars, dataset, verb=False):
    """Estimate the likelihood of a model including prior on parameters."""
    # Update model parameters factors inplace
    for factor, par in zip(pars, dataset.parameters.free_parameters):
        par.factor = factor

    lnprob_priors = lnprior(dataset)

    # dataset.likelihood returns Cash statistics values
    # emcee will maximisise the LogLikelihood so we need -dataset.likelihood
    total_lnprob = -dataset.likelihood(None) + lnprob_priors

    if verb:
        print("Parameters are:", pars)
        print("LL=", total_lnprob)
        for p in dataset.parameters.free_parameters:
            print(p)
        print("")

    return total_lnprob


# ### Define priors
# 
# This steps is a bit manual for the moment until we find a better API to define priors.<br>
# Note the you **need** to define priors for each parameter otherwise your walkers can explore uncharted territories (e.g. negative norms).

# In[ ]:


# Define the free parameters and min, max values

dataset.parameters["sigma"].frozen = True
dataset.parameters["lon_0"].frozen = True
dataset.parameters["lat_0"].frozen = True
dataset.parameters["amplitude"].frozen = False
dataset.parameters["index"].frozen = False
dataset.parameters["lambda_"].frozen = False


dataset.background_model.parameters["norm"].frozen = False
dataset.background_model.parameters["tilt"].frozen = True

dataset.background_model.parameters["norm"].min = 0.5
dataset.background_model.parameters["norm"].max = 2

dataset.parameters["index"].min = 1
dataset.parameters["index"].max = 5
dataset.parameters["lambda_"].min = 1e-3
dataset.parameters["lambda_"].max = 1

dataset.parameters["amplitude"].min = (
    0.01 * dataset.parameters["amplitude"].value
)
dataset.parameters["amplitude"].max = (
    100 * dataset.parameters["amplitude"].value
)

dataset.parameters["sigma"].min = 0.05
dataset.parameters["sigma"].max = 1

# Setting amplitude init values a bit offset to see evolution
# Here starting close to the real value
dataset.parameters["index"].value = 2.0
dataset.parameters["amplitude"].value = 3.3e-12
dataset.parameters["lambda_"].value = 0.05

print(dataset.model)
print("LL =", dataset.likelihood(dataset.parameters))


# In[ ]:


# Now let's define a function to init parameters and run the MCMC with emcee
# Depending on your number of walkers, Nrun and dimensionality, this can take a while (> minutes)

def run_mcmc(dataset, nwalkers=12, nrun=500, threads=1):
    """
    Run the MCMC sampler.
    
    Parameters
    ----------
    dataset : `MapDataset`  
        A gammapy dataset object. This contains the observed counts cube,
        the exposure cube, the psf cube, and the sky model and model.
        Each free parameter in the sky model is considered as parameter for the MCMC.
    nwalkers: int
        Required integer number of walkers to use in ensemble.
        Minimum is 2*nparam+2, but more than that is usually better.
        Must be even to use MPI mode.
    nrun: int
        Number of steps for walkers. Typically at least a few hundreds (but depends on dimensionality).
        Low nrun (<100?) will underestimate the errors. 
        Samples that would populate the distribution are nrun*nwalkers.
        This step can be ~seen as the error estimation step. 
    """
    dataset.parameters.autoscale()  # Autoscale parameters
    pars = model_to_par(dataset)  # get a tuple of parameters from dataset
    ndim = len(pars)

    # Initialize walkers in a ball of relative size 0.5% in all dimensions if the
    # parameters have been fit, or to 10% otherwise
    spread = 0.5 / 100
    p0var = np.array([spread * pp for pp in pars])
    p0 = emcee.utils.sample_ball(pars, p0var, nwalkers)

    labels = []
    for par in dataset.parameters.free_parameters:
        labels.append(par.name)
        if (par.min is np.nan) and (par.max is np.nan):
            print(
                "Warning: no priors have been set for parameter %s\n The MCMC will likely not work !"
                % (par.name)
            )

    print("List of free parameters:", labels)
    print("{} walkers will run for {} steps".format(nwalkers, nrun))
    print("Parameters init value for 1st walker:", p0[0])
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=[dataset], threads=threads
    )

    for idx, result in enumerate(sampler.sample(p0, iterations=nrun)):
        if (idx + 1) % 100 == 0:
            print("{0:5.0%}".format(idx / nrun))

    return sampler


sampler = run_mcmc(dataset, nwalkers=8, nrun=500)  # to speedup the notebook
# sampler=run_mcmc(dataset,nwalkers=16,nrun=1000) # more accurate contours


# ## Plotting the results

# ## Plot the results
# 
# The MCMC will return a sampler object containing the trace of all walkers.<br>
# The most important part is the chain attribute which is an array of shape:<br>
# _(nwalkers, nrun, nfreeparam)_
# 
# The chain is then used to plot the trace of the walkers and estimate the burnin period (the time for the walkers to reach a stationary stage).

# In[ ]:


def plot_trace(sampler, dataset):
    """
    Plot the trace of walkers for every steps
    """
    labels = []
    for par in dataset.parameters.free_parameters:
        labels.append(par.name)

    fig, ax = plt.subplots(len(labels), sharex=True)
    for i in range(len(ax)):
        ax[i].plot(sampler.chain[:, :, i].T, "-k", alpha=0.2)
        ax[i].set_ylabel(labels[i])
    plt.xlabel("Nrun")


plot_trace(sampler, dataset)


# In[ ]:


def plot_corner(sampler, dataset, nburn=0):
    """
    Corner plot for each parameter explored by the walkers.
    
    Parameters
    ----------
    sampler : `EnsembleSample`
        Sample instance.
    
    nburn: int
        Number of runs that will be discarded (burnt) until reaching ~stationary states for walkers.
        Hard to guess. Depends how close to best-fit you are. 
        A good nbrun value can be estimated from the trace plot.
        This step can be ~seen as the fitting step.    
    
    """
    labels = [par.name for par in dataset.parameters.free_parameters]

    samples = sampler.chain[:, nburn:, :].reshape((-1, len(labels)))

    corner.corner(
        samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True
    )


plot_corner(sampler, dataset, nburn=200)


# ## Plot the model dispersion
# 
# Using the samples from the chain after the burn period, we can plot the different models compared to the truth model. To do this we need to the spectral models for each parameter state in the sample.

# In[ ]:


emin, emax = [0.1, 100] * u.TeV
nburn = 300

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

for nwalk in range(0, 8):
    for n in range(nburn, nburn + 100):
        pars = sampler.chain[nwalk, n, :]

        # set model parameters
        par_to_model(dataset, pars)
        spectral_model = dataset.model.skymodels[0].spectral_model

        spectral_model.plot(
            energy_range=(emin, emax),
            ax=ax,
            energy_power=2,
            alpha=0.02,
            color="grey",
        )


sky_model_simu.spectral_model.plot(
    energy_range=(emin, emax), energy_power=2, ax=ax, color="red"
);


# ## Fun Zone
# 
# Now that you have the sampler chain, you have in your hands the entire history of each walkers in the N-Dimensional parameter space. <br>
# You can for example trace the steps of each walker in any parameter space.

# In[ ]:


# Here we plot the trace of one walker in a given parameter space
parx, pary = 0, 1

plt.plot(sampler.chain[0, :, parx], sampler.chain[0, :, pary], "ko", ms=1)
plt.plot(
    sampler.chain[0, :, parx],
    sampler.chain[0, :, pary],
    ls=":",
    color="grey",
    alpha=0.5,
)

plt.xlabel("Index")
plt.ylabel("Amplitude");


# ## PeVatrons in CTA ?
# 
# Now it's your turn to play with this MCMC notebook. For example to test the CTA performance to measure a cutoff at very high energies (100 TeV ?).
# 
# After defining your Skymodel it can be as simple as this :

# In[ ]:


# dataset = simulate_dataset(model, geom, pointing, irfs)
# sampler = run_mcmc(dataset)
# plot_trace(sampler, dataset)
# plot_corner(sampler, dataset, nburn=200)

