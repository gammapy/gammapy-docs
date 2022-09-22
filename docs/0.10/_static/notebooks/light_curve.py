
# coding: utf-8

# # Light curves
# 
# ## Introduction
# 
# This tutorial explain how to compute a light curve with Gammapy.
# 
# We will use the four Crab nebula observations from the [H.E.S.S. first public test data release](https://www.mpi-hd.mpg.de/hfm/HESS/pages/dl3-dr1/) and compute per-observation fluxes. The Crab nebula is not known to be variable at TeV energies, so we expect constant brightness within statistical and systematic errors.
# 
# The main classes we will use are:
# 
# * [gammapy.time.LightCurve](https://docs.gammapy.org/0.10/api/gammapy.time.LightCurve.html)
# * [gammapy.time.LightCurveEstimator](https://docs.gammapy.org/0.10/api/gammapy.time.LightCurveEstimator.html)
# 
# ## Setup
# 
# As usual, we'll start with some setup...

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from gammapy.utils.energy import EnergyBounds
from gammapy.data import DataStore
from gammapy.spectrum import SpectrumExtraction
from gammapy.spectrum.models import PowerLaw
from gammapy.background import ReflectedRegionsBackgroundEstimator
from gammapy.time import LightCurve, LightCurveEstimator


# ## Spectrum
# 
# The `LightCurveEstimator` is based on a 1d spectral analysis within each time bin.
# So before we can make the light curve, we have to extract 1d spectra.

# In[ ]:


data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")


# In[ ]:


mask = data_store.obs_table["TARGET_NAME"] == "Crab"
obs_ids = data_store.obs_table["OBS_ID"][mask].data
observations = data_store.get_observations(obs_ids)
print(observations)


# In[ ]:


# Target definition
target_position = SkyCoord(ra=83.63308, dec=22.01450, unit="deg")
on_region_radius = Angle("0.2 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bkg_estimator = ReflectedRegionsBackgroundEstimator(\n    on_region=on_region, observations=observations\n)\nbkg_estimator.run()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ebounds = EnergyBounds.equal_log_spacing(0.7, 100, 50, unit="TeV")\nextraction = SpectrumExtraction(\n    observations=observations,\n    bkg_estimate=bkg_estimator.result,\n    containment_correction=False,\n    e_reco=ebounds,\n    e_true=ebounds,\n)\nextraction.run()\nspectrum_observations = extraction.spectrum_observations')


# ## Light curve estimation
# 
# OK, so now that we have prepared 1D spectra (not spectral models, just the 1D counts and exposure and 2D energy dispersion matrix), we can compute a lightcurve.
# 
# To compute the light curve, a spectral model shape has to be assumed, and an energy band chosen.
# The method is then to adjust the amplitude parameter of the spectral model in each time bin to the data, resulting in a flux measurement in each time bin.

# In[ ]:


# Creat list of time bin intervals
# Here we do one time bin per observation
def time_intervals_per_obs(observations):
    for obs in observations:
        yield obs.tstart, obs.tstop


time_intervals = list(time_intervals_per_obs(observations))


# In[ ]:


# Assumed spectral model
spectral_model = PowerLaw(
    index=2, amplitude=2.0e-11 * u.Unit("1 / (cm2 s TeV)"), reference=1 * u.TeV
)


# In[ ]:


energy_range = [1, 100] * u.TeV


# In[ ]:


get_ipython().run_cell_magic('time', '', 'lc_estimator = LightCurveEstimator(extraction)\nlc = lc_estimator.light_curve(\n    time_intervals=time_intervals,\n    spectral_model=spectral_model,\n    energy_range=energy_range,\n)')


# ## Results
# 
# The light curve measurement result is stored in a table. Let's have a look at the results:

# In[ ]:


print(lc.table.colnames)


# In[ ]:


lc.table["time_min", "time_max", "flux", "flux_err"]


# In[ ]:


lc.plot();


# In[ ]:


# Let's compare to the expected flux of this source
from gammapy.spectrum import CrabSpectrum

crab_spec = CrabSpectrum().model
crab_flux = crab_spec.integral(*energy_range).to("cm-2 s-1")
crab_flux


# In[ ]:


ax = lc.plot(marker="o", lw=2)
ax.hlines(
    crab_flux.value,
    xmin=lc.table["time_min"].min(),
    xmax=lc.table["time_max"].max(),
);


# ## Exercises
# 
# * Change the assumed spectral model shape (e.g. to a steeper power-law), and see how the integral flux estimate for the lightcurve changes.
# * Try a time binning where you split the observation time for every run into two time bins.
# * Try to analyse the PKS 2155 flare data from the H.E.S.S. first public test data release.
#   Start with per-observation fluxes, and then try fluxes within 5 minute time bins for one or two of the observations where the source was very bright.
