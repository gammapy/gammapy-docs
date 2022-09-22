
# coding: utf-8

# # Spectral analysis with Gammapy

# ## Introduction
# 
# This notebook explains in detail how to use the classes in [gammapy.spectrum](https://docs.gammapy.org/0.10/spectrum/index.html) and related ones. Note, that there is also [spectrum_pipe.ipynb](spectrum_pipe.ipynb) which explains how to do the analysis using a high-level interface. This notebook is aimed at advanced users who want to script taylor-made analysis pipelines and implement new methods.
# 
# Based on a datasets of 4 Crab observations with H.E.S.S. (simulated events for now) we will perform a full region based spectral analysis, i.e. extracting source and background counts from certain 
# regions, and fitting them using the forward-folding approach. We will use the following classes
# 
# Data handling:
# 
# * [gammapy.data.DataStore](https://docs.gammapy.org/0.10/api/gammapy.data.DataStore.html)
# * [gammapy.data.DataStoreObservation](https://docs.gammapy.org/0.10/api/gammapy.data.DataStoreObservation.html)
# * [gammapy.data.ObservationStats](https://docs.gammapy.org/0.10/api/gammapy.data.ObservationStats.html)
# * [gammapy.data.ObservationSummary](https://docs.gammapy.org/0.10/api/gammapy.data.ObservationSummary.html)
# 
# To extract the 1-dim spectral information:
# 
# * [gammapy.spectrum.SpectrumObservation](https://docs.gammapy.org/0.10/api/gammapy.spectrum.SpectrumObservation.html)
# * [gammapy.spectrum.SpectrumExtraction](https://docs.gammapy.org/0.10/api/gammapy.spectrum.SpectrumExtraction.html)
# * [gammapy.background.ReflectedRegionsBackgroundEstimator](https://docs.gammapy.org/0.10/api/gammapy.background.ReflectedRegionsBackgroundEstimator.html)
# 
# 
# For the global fit (using Sherpa and WSTAT in the background):
# 
# * [gammapy.spectrum.SpectrumFit](https://docs.gammapy.org/0.10/api/gammapy.spectrum.SpectrumFit.html)
# * [gammapy.spectrum.models.PowerLaw](https://docs.gammapy.org/0.10/api/gammapy.spectrum.models.PowerLaw.html)
# * [gammapy.spectrum.models.ExponentialCutoffPowerLaw](https://docs.gammapy.org/0.10/api/gammapy.spectrum.models.ExponentialCutoffPowerLaw.html)
# * [gammapy.spectrum.models.LogParabola](https://docs.gammapy.org/0.10/api/gammapy.spectrum.models.LogParabola.html)
# 
# To compute flux points (a.k.a. "SED" = "spectral energy distribution")
# 
# * [gammapy.spectrum.SpectrumResult](https://docs.gammapy.org/0.10/api/gammapy.spectrum.SpectrumResult.html)
# * [gammapy.spectrum.FluxPoints](https://docs.gammapy.org/0.10/api/gammapy.spectrum.FluxPoints.html)
# * [gammapy.spectrum.SpectrumEnergyGroupMaker](https://docs.gammapy.org/0.10/api/gammapy.spectrum.SpectrumEnergyGroupMaker.html)
# * [gammapy.spectrum.FluxPointEstimator](https://docs.gammapy.org/0.10/api/gammapy.spectrum.FluxPointEstimator.html)
# 
# Feedback welcome!

# ## Setup
# 
# As usual, we'll start with some setup ...

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


# Check package versions
import gammapy
import numpy as np
import astropy
import regions
import sherpa

print("gammapy:", gammapy.__version__)
print("numpy:", np.__version__)
print("astropy", astropy.__version__)
print("regions", regions.__version__)
print("sherpa", sherpa.__version__)


# In[ ]:


import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.table import vstack as vstack_table
from regions import CircleSkyRegion
from gammapy.data import DataStore, Observations
from gammapy.data import ObservationStats, ObservationSummary
from gammapy.background.reflected import ReflectedRegionsBackgroundEstimator
from gammapy.utils.energy import EnergyBounds
from gammapy.spectrum import (
    SpectrumExtraction,
    SpectrumObservation,
    SpectrumFit,
    SpectrumResult,
)
from gammapy.spectrum.models import (
    PowerLaw,
    ExponentialCutoffPowerLaw,
    LogParabola,
)
from gammapy.spectrum import (
    FluxPoints,
    SpectrumEnergyGroupMaker,
    FluxPointEstimator,
)
from gammapy.maps import Map


# ## Configure logger
# 
# Most high level classes in gammapy have the possibility to turn on logging or debug output. We well configure the logger in the following. For more info see https://docs.python.org/2/howto/logging.html#logging-basic-tutorial

# In[ ]:


# Setup the logger
import logging

logging.basicConfig()
logging.getLogger("gammapy.spectrum").setLevel("WARNING")


# ## Load Data
# 
# First, we select and load some H.E.S.S. observations of the Crab nebula (simulated events for now).
# 
# We will access the events, effective area, energy dispersion, livetime and PSF for containement correction.

# In[ ]:


datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
obs_ids = [23523, 23526, 23559, 23592]
observations = datastore.get_observations(obs_ids)


# ## Define Target Region
# 
# The next step is to define a signal extraction region, also known as on region. In the simplest case this is just a [CircleSkyRegion](http://astropy-regions.readthedocs.io/en/latest/api/regions.CircleSkyRegion.html#regions.CircleSkyRegion), but here we will use the ``Target`` class in gammapy that is useful for book-keeping if you run several analysis in a script.

# In[ ]:


target_position = SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs")
on_region_radius = Angle("0.11 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)


# ## Create exclusion mask
# 
# We will use the reflected regions method to place off regions to estimate the background level in the on region.
# To make sure the off regions don't contain gamma-ray emission, we create an exclusion mask.
# 
# Using http://gamma-sky.net/ we find that there's only one known gamma-ray source near the Crab nebula: the AGN called [RGB J0521+212](http://gamma-sky.net/#/cat/tev/23) at GLON = 183.604 deg and GLAT = -8.708 deg.

# In[ ]:


exclusion_region = CircleSkyRegion(
    center=SkyCoord(183.604, -8.708, unit="deg", frame="galactic"),
    radius=0.5 * u.deg,
)

skydir = target_position.galactic
exclusion_mask = Map.create(
    npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", coordsys="GAL"
)

mask = exclusion_mask.geom.region_mask([exclusion_region], inside=False)
exclusion_mask.data = mask
exclusion_mask.plot()


# ## Estimate background
# 
# Next we will manually perform a background estimate by placing [reflected regions](https://docs.gammapy.org/0.10/background/reflected.html) around the pointing position and looking at the source statistics. This will result in a  [gammapy.background.BackgroundEstimate](https://docs.gammapy.org/0.10/api/gammapy.background.BackgroundEstimate.html) that serves as input for other classes in gammapy.

# In[ ]:


background_estimator = ReflectedRegionsBackgroundEstimator(
    observations=observations,
    on_region=on_region,
    exclusion_mask=exclusion_mask,
)

background_estimator.run()


# In[ ]:


# print(background_estimator.result[0])


# In[ ]:


plt.figure(figsize=(8, 8))
background_estimator.plot(add_legend=True)


# ## Source statistic
# 
# Next we're going to look at the overall source statistics in our signal region. For more info about what debug plots you can create check out the [ObservationSummary](https://docs.gammapy.org/0.10/api/gammapy.data.ObservationSummary.html#gammapy.data.ObservationSummary) class.

# In[ ]:


stats = []
for obs, bkg in zip(observations, background_estimator.result):
    stats.append(ObservationStats.from_observation(obs, bkg))

print(stats[1])

obs_summary = ObservationSummary(stats)
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(121)

obs_summary.plot_excess_vs_livetime(ax=ax1)
ax2 = fig.add_subplot(122)
obs_summary.plot_significance_vs_livetime(ax=ax2)


# ## Extract spectrum
# 
# Now, we're going to extract a spectrum using the [SpectrumExtraction](https://docs.gammapy.org/0.10/api/gammapy.spectrum.SpectrumExtraction.html) class. We provide the reconstructed energy binning we want to use. It is expected to be a Quantity with unit energy, i.e. an array with an energy unit. We use a utility function to create it. We also provide the true energy binning to use.

# In[ ]:


e_reco = EnergyBounds.equal_log_spacing(0.1, 40, 40, unit="TeV")
e_true = EnergyBounds.equal_log_spacing(0.05, 100.0, 200, unit="TeV")


# Instantiate a [SpectrumExtraction](https://docs.gammapy.org/0.10/api/gammapy.spectrum.SpectrumExtraction.html) object that will do the extraction. The containment_correction parameter is there to allow for PSF leakage correction if one is working with full enclosure IRFs. We also compute a threshold energy and store the result in OGIP compliant files (pha, rmf, arf). This last step might be omitted though.

# In[ ]:


ANALYSIS_DIR = "crab_analysis"

extraction = SpectrumExtraction(
    observations=observations,
    bkg_estimate=background_estimator.result,
    containment_correction=False,
)
extraction.run()

# Add a condition on correct energy range in case it is not set by default
extraction.compute_energy_threshold(method_lo="area_max", area_percent_lo=10.0)

print(extraction.spectrum_observations[0])
# Write output in the form of OGIP files: PHA, ARF, RMF, BKG
# extraction.run(observations=observations, bkg_estimate=background_estimator.result, outdir=ANALYSIS_DIR)


# ## Look at observations
# 
# Now we will look at the files we just created. We will use the [SpectrumObservation](https://docs.gammapy.org/0.10/api/gammapy.spectrum.SpectrumObservation.html) object that are still in memory from the extraction step. Note, however, that you could also read them from disk if you have written them in the step above. The ``ANALYSIS_DIR`` folder contains 4 ``FITS`` files for each observation. These files are described in detail [here](https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/ogip/index.html). In short, they correspond to the on vector, the off vector, the effectie area, and the energy dispersion.

# In[ ]:


# filename = ANALYSIS_DIR + '/ogip_data/pha_obs23523.fits'
# obs = SpectrumObservation.read(filename)

# Requires IPython widgets
# _ = extraction.spectrum_observations.peek()

extraction.spectrum_observations[0].peek()


# ## Fit spectrum
# 
# Now we'll fit a global model to the spectrum. First we do a joint likelihood fit to all observations. If you want to stack the observations see below. We will also produce a debug plot in order to show how the global fit matches one of the individual observations.

# In[ ]:


model = PowerLaw(
    index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
)

joint_fit = SpectrumFit(obs_list=extraction.spectrum_observations, model=model)
joint_fit.run()
joint_result = joint_fit.result


# In[ ]:


ax0, ax1 = joint_result[0].plot(figsize=(8, 8))
ax0.set_ylim(0, 20)
print(joint_result[0])


# ## Compute Flux Points
# 
# To round up our analysis we can compute flux points by fitting the norm of the global model in energy bands. We'll use a fixed energy binning for now.

# In[ ]:


# Define energy binning
stacked_obs = extraction.spectrum_observations.stack()

e_min, e_max = stacked_obs.lo_threshold.to_value("TeV"), 30
ebounds = np.logspace(np.log10(e_min), np.log10(e_max), 15) * u.TeV


seg = SpectrumEnergyGroupMaker(obs=stacked_obs)
seg.compute_groups_fixed(ebounds=ebounds)

print(seg.groups)


# In[ ]:


fpe = FluxPointEstimator(
    obs=stacked_obs, groups=seg.groups, model=joint_result[0].model
)
flux_points = fpe.run()


# In[ ]:


flux_points.table_formatted


# Now we plot the flux points and their likelihood profiles. For the plotting of upper limits we choose a threshold of TS < 4.

# In[ ]:


flux_points.table["is_ul"] = flux_points.table["ts"] < 4
ax = flux_points.plot(
    energy_power=2, flux_unit="erg-1 cm-2 s-1", color="darkorange"
)
flux_points.to_sed_type("e2dnde").plot_likelihood(ax=ax)


# The final plot with the best fit model, flux points and residuals can be quickly made like this

# In[ ]:


spectrum_result = SpectrumResult(
    points=flux_points, model=joint_result[0].model
)
ax0, ax1 = spectrum_result.plot(
    energy_range=joint_fit.result[0].fit_range,
    energy_power=2,
    flux_unit="erg-1 cm-2 s-1",
    fig_kwargs=dict(figsize=(8, 8)),
)

ax0.set_xlim(0.4, 50)


# ## Stack observations
# 
# And alternative approach to fitting the spectrum is stacking all observations first and the fitting a model to the stacked observation. This works as follows. A comparison to the joint likelihood fit is also printed.

# In[ ]:


stacked_obs = extraction.spectrum_observations.stack()

stacked_fit = SpectrumFit(obs_list=stacked_obs, model=model)
stacked_fit.run()


# In[ ]:


stacked_result = stacked_fit.result
print(stacked_result[0])


# In[ ]:


stacked_table = stacked_result[0].to_table(format=".3g")
stacked_table["method"] = "stacked"
joint_table = joint_result[0].to_table(format=".3g")
joint_table["method"] = "joint"
total_table = vstack_table([stacked_table, joint_table])
print(
    total_table["method", "index", "index_err", "amplitude", "amplitude_err"]
)


# ## Exercises
# 
# Some things we might do:
# 
# - Fit a different spectral model (ECPL or CPL or ...)
# - Use different method or parameters to compute the flux points
# - Do a chi^2 fit to the flux points and compare
# 
# TODO: give pointers how to do this (and maybe write a notebook with solutions)

# In[ ]:


# Start exercises here

