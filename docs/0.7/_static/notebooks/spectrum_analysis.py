
# coding: utf-8

# # Spectral analysis with Gammapy

# ## Introduction
# 
# This notebook explains in detail how to use the classes in [gammapy.spectrum](http://docs.gammapy.org/0.7/spectrum/index.html) and related ones. Note, that there is also [spectrum_pipe.ipynb](spectrum_pipe.ipynb) which explains how to do the analysis using a high-level interface. This notebook is aimed at advanced users who want to script taylor-made analysis pipelines and implement new methods.
# 
# Based on a datasets of 4 Crab observations with H.E.S.S. (simulated events for now) we will perform a full region based spectral analysis, i.e. extracting source and background counts from certain 
# regions, and fitting them using the forward-folding approach. We will use the following classes
# 
# Data handling:
# 
# * [gammapy.data.DataStore](http://docs.gammapy.org/0.7/api/gammapy.data.DataStore.html)
# * [gammapy.data.DataStoreObservation](http://docs.gammapy.org/0.7/api/gammapy.data.DataStoreObservation.html)
# * [gammapy.data.ObservationStats](http://docs.gammapy.org/0.7/api/gammapy.data.ObservationStats.html)
# * [gammapy.data.ObservationSummary](http://docs.gammapy.org/0.7/api/gammapy.data.ObservationSummary.html)
# 
# To extract the 1-dim spectral information:
# 
# * [gammapy.spectrum.SpectrumObservation](http://docs.gammapy.org/0.7/api/gammapy.spectrum.SpectrumObservation.html)
# * [gammapy.spectrum.SpectrumExtraction](http://docs.gammapy.org/0.7/api/gammapy.spectrum.SpectrumExtraction.html)
# * [gammapy.background.ReflectedRegionsBackgroundEstimator](http://docs.gammapy.org/0.7/api/gammapy.background.ReflectedRegionsBackgroundEstimator.html)
# 
# 
# For the global fit (using Sherpa and WSTAT in the background):
# 
# * [gammapy.spectrum.SpectrumFit](http://docs.gammapy.org/0.7/api/gammapy.spectrum.SpectrumFit.html)
# * [gammapy.spectrum.models.PowerLaw](http://docs.gammapy.org/0.7/api/gammapy.spectrum.models.PowerLaw.html)
# * [gammapy.spectrum.models.ExponentialCutoffPowerLaw](http://docs.gammapy.org/0.7/api/gammapy.spectrum.models.ExponentialCutoffPowerLaw.html)
# * [gammapy.spectrum.models.LogParabola](http://docs.gammapy.org/0.7/api/gammapy.spectrum.models.LogParabola.html)
# 
# To compute flux points (a.k.a. "SED" = "spectral energy distribution")
# 
# * [gammapy.spectrum.SpectrumResult](http://docs.gammapy.org/0.7/api/gammapy.spectrum.SpectrumResult.html)
# * [gammapy.spectrum.FluxPoints](http://docs.gammapy.org/0.7/api/gammapy.spectrum.FluxPoints.html)
# * [gammapy.spectrum.SpectrumEnergyGroupMaker](http://docs.gammapy.org/0.7/api/gammapy.spectrum.SpectrumEnergyGroupMaker.html)
# * [gammapy.spectrum.FluxPointEstimator](http://docs.gammapy.org/0.7/api/gammapy.spectrum.FluxPointEstimator.html)
# 
# Feedback welcome!

# ## Setup
# 
# As usual, we'll start with some setup ...

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


# Check package versions
import gammapy
import numpy as np
import astropy
import regions
import sherpa

print('gammapy:', gammapy.__version__)
print('numpy:', np.__version__)
print('astropy', astropy.__version__)
print('regions', regions.__version__)
print('sherpa', sherpa.__version__)


# In[3]:


import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.table import vstack as vstack_table
from regions import CircleSkyRegion
from gammapy.data import DataStore, ObservationList
from gammapy.data import ObservationStats, ObservationSummary
from gammapy.background.reflected import ReflectedRegionsBackgroundEstimator
from gammapy.utils.energy import EnergyBounds
from gammapy.spectrum import SpectrumExtraction, SpectrumObservation, SpectrumFit, SpectrumResult
from gammapy.spectrum.models import PowerLaw, ExponentialCutoffPowerLaw, LogParabola
from gammapy.spectrum import FluxPoints, SpectrumEnergyGroupMaker, FluxPointEstimator
from gammapy.image import SkyImage


# ## Configure logger
# 
# Most high level classes in gammapy have the possibility to turn on logging or debug output. We well configure the logger in the following. For more info see https://docs.python.org/2/howto/logging.html#logging-basic-tutorial

# In[4]:


# Setup the logger
import logging
logging.basicConfig()
log = logging.getLogger('gammapy.spectrum')
log.setLevel(logging.WARNING)


# ## Load Data
# 
# First, we select and load some H.E.S.S. observations of the Crab nebula (simulated events for now).
# 
# We will access the events, effective area, energy dispersion, livetime and PSF for containement correction.

# In[5]:


DATA_DIR = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2'

datastore = DataStore.from_dir(DATA_DIR)
obs_ids = [23523, 23526, 23559, 23592]

obs_list = datastore.obs_list(obs_ids)


# ## Define Target Region
# 
# The next step is to define a signal extraction region, also known as on region. In the simplest case this is just a [CircleSkyRegion](http://astropy-regions.readthedocs.io/en/latest/api/regions.CircleSkyRegion.html#regions.CircleSkyRegion), but here we will use the ``Target`` class in gammapy that is useful for book-keeping if you run several analysis in a script.

# In[6]:


target_position = SkyCoord(ra=83.63, dec=22.01, unit='deg', frame='icrs')
on_region_radius = Angle('0.11 deg')
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)


# ## Load exclusion mask
# 
# Most analysis will require a mask to exclude regions with possible gamma-ray signal from the background estimation procedure. For simplicity, we will use a pre-cooked exclusion mask from gammapy-extra which includes (or rather excludes) all source listed in the [TeVCat](http://tevcat.uchicago.edu/) and cutout only the region around the crab.
# 
# TODO: Change to [gamma-cat](https://gammapy.github.io/gamma-cat/)

# In[7]:


EXCLUSION_FILE = '$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits'

allsky_mask = SkyImage.read(EXCLUSION_FILE)
exclusion_mask = allsky_mask.cutout(
    position=on_region.center,
    size=Angle('6 deg'),
)


# ## Estimate background
# 
# Next we will manually perform a background estimate by placing [reflected regions](http://docs.gammapy.org/0.7/background/reflected.html) around the pointing position and looking at the source statistics. This will result in a  [gammapy.background.BackgroundEstimate](http://docs.gammapy.org/0.7/api/gammapy.background.BackgroundEstimate.html) that serves as input for other classes in gammapy.

# In[8]:


background_estimator = ReflectedRegionsBackgroundEstimator(
    obs_list=obs_list,
    on_region=on_region,
    exclusion_mask = exclusion_mask)

background_estimator.run()
print(background_estimator.result[0])


# In[9]:


plt.figure(figsize=(8,8))
background_estimator.plot()


# ## Source statistic
# 
# Next we're going to look at the overall source statistics in our signal region. For more info about what debug plots you can create check out the [ObservationSummary](http://docs.gammapy.org/0.7/api/gammapy.data.ObservationSummary.html#gammapy.data.ObservationSummary) class.

# In[10]:


stats = []
for obs, bkg in zip(obs_list, background_estimator.result):
    stats.append(ObservationStats.from_obs(obs, bkg))
    
print(stats[1])

obs_summary = ObservationSummary(stats)
fig = plt.figure(figsize=(10,6))
ax1=fig.add_subplot(121)

obs_summary.plot_excess_vs_livetime(ax=ax1)
ax2=fig.add_subplot(122)
obs_summary.plot_significance_vs_livetime(ax=ax2)


# ## Extract spectrum
# 
# Now, we're going to extract a spectrum using the [SpectrumExtraction](http://docs.gammapy.org/0.7/api/gammapy.spectrum.SpectrumExtraction.html) class. We provide the reconstructed energy binning we want to use. It is expected to be a Quantity with unit energy, i.e. an array with an energy unit. We use a utility function to create it. We also provide the true energy binning to use.

# In[11]:


e_reco = EnergyBounds.equal_log_spacing(0.1, 40, 40, unit='TeV')
e_true = EnergyBounds.equal_log_spacing(0.05, 100., 200, unit='TeV')


# Instantiate a [SpectrumExtraction](http://docs.gammapy.org/0.7/api/gammapy.spectrum.SpectrumExtraction.html) object that will do the extraction. The containment_correction parameter is there to allow for PSF leakage correction if one is working with full enclosure IRFs. We also compute a threshold energy and store the result in OGIP compliant files (pha, rmf, arf). This last step might be omitted though.

# In[12]:


ANALYSIS_DIR = 'crab_analysis'

extraction = SpectrumExtraction(
    obs_list=obs_list,
    bkg_estimate=background_estimator.result,
    containment_correction=False,
)
extraction.run()

# Add a condition on correct energy range in case it is not set by default
extraction.compute_energy_threshold(method_lo='area_max', area_percent_lo=10.0)

print(extraction.observations[0])
# Write output in the form of OGIP files: PHA, ARF, RMF, BKG
# extraction.run(obs_list=obs_list, bkg_estimate=background_estimator.result, outdir=ANALYSIS_DIR)


# ## Look at observations
# 
# Now we will look at the files we just created. We will use the [SpectrumObservation](http://docs.gammapy.org/0.7/api/gammapy.spectrum.SpectrumObservation.html) object that are still in memory from the extraction step. Note, however, that you could also read them from disk if you have written them in the step above . The ``ANALYSIS_DIR`` folder contains 4 ``FITS`` files for each observation. These files are described in detail at https://gamma-astro-data-formats.readthedocs.io/en/latest/ogip/index.html. In short they correspond to the on vector, the off vector, the effectie area, and the energy dispersion.

# In[13]:


#filename = ANALYSIS_DIR + '/ogip_data/pha_obs23523.fits'
#obs = SpectrumObservation.read(filename)

# Requires IPython widgets
#_ = extraction.observations.peek()

extraction.observations[0].peek()


# ## Fit spectrum
# 
# Now we'll fit a global model to the spectrum. First we do a joint likelihood fit to all observations. If you want to stack the observations see below. We will also produce a debug plot in order to show how the global fit matches one of the individual observations.

# In[14]:


model = PowerLaw(
    index=2 * u.Unit(''),
    amplitude=2e-11 * u.Unit('cm-2 s-1 TeV-1'),
    reference=1 * u.TeV,
)

joint_fit = SpectrumFit(obs_list=extraction.observations, model=model)

joint_fit.fit()
joint_fit.est_errors()
#fit.run(outdir = ANALYSIS_DIR)

joint_result = joint_fit.result


# In[15]:


ax0, ax1 = joint_result[0].plot(figsize=(8,8))
ax0.set_ylim(0, 20)
print(joint_result[0])


# ## Compute Flux Points
# 
# To round up out analysis we can compute flux points by fitting the norm of the global model in energy bands. We'll use a fixed energy binning for now.

# In[16]:


# Define energy binning
ebounds = [0.3, 1.1, 3, 10.1, 30] * u.TeV

stacked_obs = extraction.observations.stack()

seg = SpectrumEnergyGroupMaker(obs=stacked_obs)
seg.compute_range_safe()
seg.compute_groups_fixed(ebounds=ebounds)

print(seg.groups)


# In[17]:


fpe = FluxPointEstimator(
    obs=stacked_obs,
    groups=seg.groups,
    model=joint_result[0].model,
)
fpe.compute_points()


# In[18]:


fpe.flux_points.plot()
fpe.flux_points.table


# The final plot with the best fit model and the flux points can be quickly made like this

# In[19]:


spectrum_result = SpectrumResult(
    points=fpe.flux_points,
    model=joint_result[0].model,
)
ax0, ax1 = spectrum_result.plot(
    energy_range=joint_fit.result[0].fit_range,
    energy_power=2, flux_unit='erg-1 cm-2 s-1',
    fig_kwargs=dict(figsize=(8,8)),
    point_kwargs=dict(color='navy')
)

ax0.set_xlim(0.4, 50)


# ## Stack observations
# 
# And alternative approach to fitting the spectrum is stacking all observations first and the fitting a model to the stacked observation. This works as follows. A comparison to the joint likelihood fit is also printed.

# In[20]:


stacked_obs = extraction.observations.stack()

stacked_fit = SpectrumFit(obs_list=stacked_obs, model=model)
stacked_fit.fit()
stacked_fit.est_errors()


stacked_result = stacked_fit.result
print(stacked_result[0])

stacked_table = stacked_result[0].to_table(format='.3g')
stacked_table['method'] = 'stacked'
joint_table = joint_result[0].to_table(format='.3g')
joint_table['method'] = 'joint'
total_table = vstack_table([stacked_table, joint_table])
print(total_table['method', 'index', 'index_err', 'amplitude', 'amplitude_err'])


# ## Exercises
# 
# Some things we might do:
# 
# - Fit a different spectral model (ECPL or CPL or ...)
# - Use different method or parameters to compute the flux points
# - Do a chi^2 fit to the flux points and compare
# 
# TODO: give pointers how to do this (and maybe write a notebook with solutions)

# In[21]:


# Start exercises here


# ## What next?
# 
# In this tutorial we learned how to extract counts spectra from an event list and generate the corresponding IRFs. Then we fitted a model to the observations and also computed flux points.
# 
# Here's some suggestions where to go next:
# 
# * if you want think this is way too complicated and just want to run a quick analysis check out [this notebook](spectrum_pipe.ipynb)
# * if you interested in available fit statistics checkout [gammapy.stats](http://docs.gammapy.org/0.7/stats/index.html)
# * if you want to simulate spectral look at [this tutorial](http://docs.gammapy.org/0.7/spectrum/simulation.html)
# * if you want to compare your spectra to e.g. Fermi spectra published in catalogs have a look at [this](http://docs.gammapy.org/0.7/spectrum/plotting_fermi_spectra.html)
