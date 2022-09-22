#!/usr/bin/env python
# coding: utf-8

# # Modeling and fitting
# 
# 
# ## Prerequisites
# 
# - Knowledge of spectral analysis to produce 1D On-Off datasets, [see the following tutorial](spectrum_analysis.ipynb)
# - Reading of pre-computed datasets [see the MWL tutorial](analysis_mwl.ipynb)
# - General knowledge on statistics and optimization methods
# 
# ## Proposed approach
# 
# This is a hands-on tutorial to `~gammapy.modeling`, showing how the model, dataset and fit classes work together. As an example we are going to work with HESS data of the Crab Nebula and show in particular how to :
# - perform a spectral analysis
# - use different fitting backends
# - acces covariance matrix informations and parameter errors
# - compute likelihood profile
# - compute confidence contours
# 
# See also: [Models gallery tutorial](models.ipynb) and `docs/modeling/index.rst`.
# 
# 
# ## The setup

# In[ ]:


import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import scipy.stats as st
from gammapy.modeling import Fit
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.modeling.models import LogParabolaSpectralModel, SkyModel
from gammapy.visualization.utils import plot_contour_line
from itertools import combinations


# ## Model and dataset
# 
# First we define the source model, here we need only a spectral model for which we choose a log-parabola

# In[ ]:


crab_spectrum = LogParabolaSpectralModel(
    amplitude=1e-11 / u.cm ** 2 / u.s / u.TeV,
    reference=1 * u.TeV,
    alpha=2.3,
    beta=0.2,
)

crab_spectrum.alpha.max = 3
crab_spectrum.alpha.min = 1
crab_model = SkyModel(spectral_model=crab_spectrum, name="crab")


# The data and background are read from pre-computed ON/OFF datasets of HESS observations, for simplicity we stack them together.
# Then we set the model and fit range to the resulting dataset.

# In[ ]:


datasets = []
for obs_id in [23523, 23526]:
    dataset = SpectrumDatasetOnOff.from_ogip_files(
        f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obs_id}.fits"
    )
    datasets.append(dataset)

dataset_hess = Datasets(datasets).stack_reduce(name="HESS")

# Set model and fit range
dataset_hess.models = crab_model
e_min = 0.66 * u.TeV
e_max = 30 * u.TeV
dataset_hess.mask_fit = dataset_hess.counts.geom.energy_mask(e_min, e_max)


# ## Fitting options
# 
# 
# 
# First let's create a `Fit` instance:

# In[ ]:


fit = Fit([dataset_hess], store_trace=True)


# By default the fit is performed using MINUIT, you can select alternative optimizers and set their option using the `optimize_opts` argument of the `Fit.run()` method. In addition we have specified to store the trace of parameter values of the fit.
# 
# Note that, for now, covaraince matrix and errors are computed only for the fitting with MINUIT. However depending on the problem other optimizers can better perform, so somethimes it can be usefull to run a pre-fit with alternative optimization methods.
# 
# For the "scipy" backend the available options are desribed in detail here:  
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

# In[ ]:


get_ipython().run_cell_magic('time', '', 'scipy_opts = {"method": "L-BFGS-B", "options": {"ftol": 1e-4, "gtol": 1e-05}}\nresult_scipy = fit.run(backend="scipy", optimize_opts=scipy_opts)')


# For the "sherpa" backend you can choose the optimization algorithm between method = {"simplex",  "levmar", "moncar", "gridsearch"}.  
# Those methods are described and compared in detail on http://cxc.cfa.harvard.edu/sherpa/methods/index.html.  
# The available options of the optimization methods are described on the following page https://cxc.cfa.harvard.edu/sherpa/methods/opt_methods.html

# In[ ]:


get_ipython().run_cell_magic('time', '', 'sherpa_opts = {"method": "simplex", "ftol": 1e-3, "maxfev": int(1e4)}\nresults_simplex = fit.run(backend="sherpa", optimize_opts=sherpa_opts)')


# For the "minuit" backend see https://iminuit.readthedocs.io/en/latest/reference.html for a detailed description of the available options. If there is an entry ‘migrad_opts’, those options will be passed to [iminuit.Minuit.migrad](https://iminuit.readthedocs.io/en/latest/reference.html#iminuit.Minuit.migrad). Additionnaly you can set the fit tolerance using the [tol](https://iminuit.readthedocs.io/en/latest/reference.html#iminuit.Minuit.tol
# ) option. The minimization will stop when the estimated distance to the minimum is less than 0.001*tol (by default tol=0.1). The [strategy](https://iminuit.readthedocs.io/en/latest/reference.html#iminuit.Minuit.strategy) option change the speed and accuracy of the optimizer: 0 fast, 1 default, 2 slow but accurate. If you want more reliable error estimates, you should run the final fit with strategy 2.
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'minuit_opts = {"tol": 0.001, "strategy": 1}\nresult_minuit = fit.run(backend="minuit", optimize_opts=minuit_opts)')


# ## Fit quality assessment
# 
# There are various ways to check the convergence and quality of a fit. Among them:
# 
# - Refer to the automatically-generated results dictionary

# In[ ]:


print(result_scipy)


# In[ ]:


print(results_simplex)


# In[ ]:


print(result_minuit)


# - Check the trace of the fit e.g. in case the fit did not converge properly

# In[ ]:


result_minuit.trace


# - Check that the fitted values and errors for all parameters are reasonable, and no fitted parameter value is "too close" - or even outside - its allowed min-max range

# In[ ]:


result_minuit.parameters.to_table()


# - Plot fit statistic profiles for all fitted prameters, using `~gammapy.modeling.Fit.stat_profile()`. For a good fit and error estimate each profile should be parabolic

# In[ ]:


total_stat = result_minuit.total_stat

for par in dataset_hess.models.parameters:
    if par.frozen is False:
        profile = fit.stat_profile(parameter=par)
        plt.plot(
            profile[f"{par.name}_scan"], profile["stat_scan"] - total_stat
        )
        plt.xlabel(f"{par.unit}")
        plt.ylabel("Delta TS")
        plt.title(f"{par.name}: {par.value} +- {par.error}")
        plt.show()
        plt.close()


# - Inspect model residuals. Those can always be accessed using `~Dataset.residuals()`, that will return an array in case a the fitted `Dataset` is a `SpectrumDataset` and a full cube in case of a `MapDataset`. For more details, we refer here to the dedicated fitting tutorials: [analysis_3d.ipynb](analysis_3d.ipynb) (for `MapDataset` fitting) and [spectrum_analysis.ipynb](spectrum_analysis.ipynb) (for `SpectrumDataset` fitting).

# ## Covariance and parameters errors
# 
# After the fit the covariance matrix is attached to the model. You can get the error on a specific parameter by accessing the `.error` attribute:

# In[ ]:


crab_model.spectral_model.alpha.error


# As an example, this step is needed to produce a butterfly plot showing the envelope of the model taking into account parameter uncertainties.

# In[ ]:


energy_range = [1, 10] * u.TeV
crab_spectrum.plot(energy_range=energy_range, energy_power=2)
ax = crab_spectrum.plot_error(energy_range=energy_range, energy_power=2)


# ## Confidence contours
# 
# 
# In most studies, one wishes to estimate parameters distribution using observed sample data.
# A 1-dimensional confidence interval gives an estimated range of values which is likely to include an unknown parameter.
# A confidence contour is a 2-dimensional generalization of a confidence interval, often represented as an ellipsoid around the best-fit value.
# 
# Gammapy offers two ways of computing confidence contours, in the dedicated methods `Fit.minos_contour()` and `Fit.stat_profile()`. In the following sections we will describe them.

# An important point to keep in mind is: *what does a $N\sigma$ confidence contour really mean?* The answer is it represents the points of the parameter space for which the model likelihood is $N\sigma$ above the minimum. But one always has to keep in mind that **1 standard deviation in two dimensions has a smaller coverage probability than 68%**, and similarly for all other levels. In particular, in 2-dimensions the probability enclosed by the $N\sigma$ confidence contour is $P(N)=1-e^{-N^2/2}$.

# ### Computing contours using `Fit.minos_contour()` 

# After the fit, MINUIT offers the possibility to compute the confidence confours.
# gammapy provides an interface to this functionnality throught the `Fit` object using the `minos_contour` method.
# Here we defined a function to automatize the contour production for the differents parameterer and confidence levels (expressed in term of sigma):

# In[ ]:


def make_contours(fit, result, npoints, sigmas):
    cts_sigma = []
    for sigma in sigmas:
        contours = dict()
        for par_1, par_2 in combinations(["alpha", "beta", "amplitude"], r=2):
            contour = fit.minos_contour(
                result.parameters[par_1],
                result.parameters[par_2],
                numpoints=npoints,
                sigma=sigma,
            )
            contours[f"contour_{par_1}_{par_2}"] = {
                par_1: contour[par_1].tolist(),
                par_2: contour[par_2].tolist(),
            }
        cts_sigma.append(contours)
    return cts_sigma


# Now we can compute few contours.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'sigma = [1, 2]\ncts_sigma = make_contours(fit, result_minuit, 10, sigma)')


# Then we prepare some aliases and annotations in order to make the plotting nicer.

# In[ ]:


pars = {
    "phi": r"$\phi_0 \,/\,(10^{-11}\,{\rm TeV}^{-1} \, {\rm cm}^{-2} {\rm s}^{-1})$",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
}

panels = [
    {
        "x": "alpha",
        "y": "phi",
        "cx": (lambda ct: ct["contour_alpha_amplitude"]["alpha"]),
        "cy": (
            lambda ct: np.array(1e11)
            * ct["contour_alpha_amplitude"]["amplitude"]
        ),
    },
    {
        "x": "beta",
        "y": "phi",
        "cx": (lambda ct: ct["contour_beta_amplitude"]["beta"]),
        "cy": (
            lambda ct: np.array(1e11)
            * ct["contour_beta_amplitude"]["amplitude"]
        ),
    },
    {
        "x": "alpha",
        "y": "beta",
        "cx": (lambda ct: ct["contour_alpha_beta"]["alpha"]),
        "cy": (lambda ct: ct["contour_alpha_beta"]["beta"]),
    },
]


# Finally we produce the confidence contours figures.

# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ["m", "b", "c"]
for p, ax in zip(panels, axes):
    xlabel = pars[p["x"]]
    ylabel = pars[p["y"]]
    for ks in range(len(cts_sigma)):
        plot_contour_line(
            ax,
            p["cx"](cts_sigma[ks]),
            p["cy"](cts_sigma[ks]),
            lw=2.5,
            color=colors[ks],
            label=f"{sigma[ks]}" + r"$\sigma$",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
plt.legend()
plt.tight_layout()


# ### Computing contours using `Fit.stat_surface()`

# This alternative method for the computation of confidence contours, although more time consuming than `Fit.minos_contour()`, is expected to be more stable. It consists of a generalization of `Fit.stat_profile()` to a 2-dimensional parameter space. The algorithm is very simple:
# - First, passing two arrays of parameters values, a 2-dimensional discrete parameter space is defined;
# - For each node of the parameter space, the two parameters of interest are frozen. This way, a likelihood value ($-2\mathrm{ln}\,\mathcal{L}$, actually) is computed, by either freezing (default) or fitting all nuisance parameters;
# - Finally, a 2-dimensional surface of $-2\mathrm{ln}(\mathcal{L})$ values is returned.
# Using that surface, one can easily compute a surface of $TS = -2\Delta\mathrm{ln}(\mathcal{L})$ and compute confidence contours.
# 
# Let's see it step by step.

# First of all, we can notice that this method is "backend-agnostic", meaning that it can be run with MINUIT, sherpa or scipy as fitting tools. Here we will stick with MINUIT, which is the default choice:

# In[ ]:


optimize_opts = {"backend": "minuit", "print_level": 0}


# As an example, we can compute the confidence contour for the `alpha` and `beta` parameters of the `dataset_hess`. Here we define the parameter space:

# In[ ]:


result = result_minuit
par_1 = result.parameters["alpha"]
par_2 = result.parameters["beta"]

x = par_1
y = par_2
x_values = np.linspace(1.55, 2.7, 20)
y_values = np.linspace(-0.05, 0.55, 20)


# Then we run the algorithm, by choosing `reoptimize=False` for the sake of time saving. In real life applications, we strongly recommend to  use `reoptimize=True`, so that all free nuisance parameters will be fit at each grid node. This is the correct way, statistically speaking, of computing confidence contours, but is expected to be time consuming.

# In[ ]:


stat_surface = fit.stat_surface(
    x, y, x_values, y_values, reoptimize=False, **optimize_opts
)


# In order to easily inspect the results, we can convert the $-2\mathrm{ln}(\mathcal{L})$ surface to a surface of statistical significance (in units of Gaussian standard deviations from the surface minimum):

# In[ ]:


# Compute TS
TS = stat_surface["stat_scan"] - result.total_stat


# In[ ]:


# Compute the corresponding statistical significance surface
gaussian_sigmas = np.sqrt(TS.T)


# Notice that, as explained before, $1\sigma$ contour obtained this way will not contain 68% of the probability, but rather 

# In[ ]:


# Compute the corresponding statistical significance surface
# p_value = 1 - st.chi2(df=1).cdf(TS)
# gaussian_sigmas = st.norm.isf(p_value / 2).T


# Finally, we can plot the surface values together with contours:

# In[ ]:


fig, ax = plt.subplots(figsize=(8, 6))

# We choose to plot 1 and 2 sigma confidence contours
levels = [1, 2]

contours = plt.contour(gaussian_sigmas, levels=levels, colors="white")
plt.clabel(contours, fmt="%.0f$\,\sigma$", inline=3, fontsize=15)

im = plt.imshow(
    gaussian_sigmas,
    extent=[0, len(x_values) - 1, 0, len(y_values) - 1],
    origin="lower",
)
fig.colorbar(im)

plt.xticks(range(len(x_values)), np.around(x_values, decimals=2), rotation=45)
plt.yticks(range(len(y_values)), np.around(y_values, decimals=2));


# Note that, if computed with `reoptimize=True`, this plot would be completely consistent with the third panel of the plot produced with `Fit.minos_contour` (try!).

# Finally, it is always remember that confidence contours are approximations. In particular, when the parameter range boundaries are close to the contours lines, it is expected that the statistical meaning of the countours is not well defined. That's why we advise to always choose a parameter space that com contain the contours you're interested in.

# In[ ]:




