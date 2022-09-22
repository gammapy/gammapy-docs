#!/usr/bin/env python
# coding: utf-8

# # Pulsar analysis with Gammapy

# ## Introduction

# This notebook shows how to do a pulsar analysis with Gammapy. It's based on a Vela simulation file from the CTA DC1, which already contains a column of phases. We will produce a phasogram, a phase-resolved map and a phase-resolved spectrum of the Vela pulsar using the class PhaseBackgroundEstimator from gammapy.background.phase. 
# 
# The phasing in itself is not done here, and it requires specific packages like Tempo2 or PINT (https://nanograv-pint.readthedocs.io/en/latest/readme.html).

# ## Opening the data

# Let's first do the imports and load the only observation containing Vela in the CTA 1DC dataset shipped with Gammapy.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


from gammapy.utils.regions import SphericalCircleSkyRegion
from astropy.coordinates import SkyCoord
import astropy.units as u

from gammapy.maps import Map, WcsGeom
from gammapy.cube import fill_map_counts
from gammapy.data import DataStore
from gammapy.background import PhaseBackgroundEstimator
from gammapy.spectrum.models import PowerLaw
from gammapy.utils.fitting import Fit
from gammapy.spectrum import (
    SpectrumExtraction,
    FluxPointsEstimator,
    FluxPointsDataset,
)


# Load the data store (which is a subset of CTA-DC1 data):

# In[ ]:


data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps")


# Define obsevation ID and print events:

# In[ ]:


id_obs_vela = [111630]
obs_list_vela = data_store.get_observations(id_obs_vela)
print(obs_list_vela[0].events)


# Now that we have our observation, let's select the events in 0.2° radius around the pulsar position.

# In[ ]:


pos_target = SkyCoord(ra=128.836 * u.deg, dec=-45.176 * u.deg, frame="icrs")
on_radius = 0.2 * u.deg
on_region = SphericalCircleSkyRegion(pos_target, on_radius)

# Apply angular selection
events_vela = obs_list_vela[0].events.select_region(on_region)
print(events_vela)


# Let's load the phases of the selected events in a dedicated array.

# In[ ]:


phases = events_vela.table["PHASE"]

# Let's take a look at the first 10 phases
phases[:10]


# ## Phasogram

# Once we have the phases, we can make a phasogram. A phasogram is a histogram of phases and it works exactly like any other histogram (you can set the binning, evaluate the errors based on the counts in each bin, etc).

# In[ ]:


nbins = 30
phase_min, phase_max = (0, 1)
values, bin_edges = np.histogram(
    phases, range=(phase_min, phase_max), bins=nbins
)
bin_width = (phase_max - phase_min) / nbins

bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2


# Poissonian uncertainty on each bin
values_err = np.sqrt(values)


# In[ ]:


plt.bar(
    x=bin_center,
    height=values,
    width=bin_width,
    color="#d53d12",
    alpha=0.8,
    edgecolor="black",
    yerr=values_err,
)
plt.xlim(0, 1)
plt.xlabel("Phase")
plt.ylabel("Counts")
plt.title("Phaseogram with angular cut of {}".format(on_radius));


# Now let's add some fancy additions to our phasogram: a patch on the ON- and OFF-phase regions and one for the background level.

# In[ ]:


# Evaluate background level
off_phase_range = (0.7, 1.0)
on_phase_range = (0.5, 0.6)

mask_off = (off_phase_range[0] < phases) & (phases < off_phase_range[1])

count_bkg = mask_off.sum()
print("Number of Off events: {}".format(count_bkg))


# In[ ]:


# bkg level normalized by the size of the OFF zone (0.3)
bkg = count_bkg / nbins / (off_phase_range[1] - off_phase_range[0])

# error on the background estimation
bkg_err = (
    np.sqrt(count_bkg) / nbins / (off_phase_range[1] - off_phase_range[0])
)


# In[ ]:


# Let's redo the same plot for the basis
plt.bar(
    x=bin_center,
    height=values,
    width=bin_width,
    color="#d53d12",
    alpha=0.8,
    edgecolor="black",
    yerr=values_err,
)

# Plot background level
x_bkg = np.linspace(0, 1, 50)

kwargs = {"color": "black", "alpha": 0.5, "ls": "--", "lw": 2}

plt.plot(x_bkg, (bkg - bkg_err) * np.ones_like(x_bkg), **kwargs)
plt.plot(x_bkg, (bkg + bkg_err) * np.ones_like(x_bkg), **kwargs)

plt.fill_between(
    x_bkg, bkg - bkg_err, bkg + bkg_err, facecolor="grey", alpha=0.5
)  # grey area for the background level

# Let's make patches for the on and off phase zones
on_patch = plt.axvspan(
    on_phase_range[0], on_phase_range[1], alpha=0.3, color="gray", ec="black"
)

off_patch = plt.axvspan(
    off_phase_range[0],
    off_phase_range[1],
    alpha=0.4,
    color="white",
    hatch="x",
    ec="black",
)

# Legends "ON" and "OFF"
plt.text(0.55, 5, "ON", color="black", fontsize=17, ha="center")
plt.text(0.895, 5, "OFF", color="black", fontsize=17, ha="center")
plt.xlabel("Phase")
plt.ylabel("Counts")
plt.xlim(0, 1)
plt.title("Phasogram with angular cut of {}".format(on_radius));


# ## Phase-resolved map

# Now that the phases are computed, we want to do a phase-resolved sky map : a map of the ON-phase events minus alpha times the OFF-phase events. Alpha is the ratio between the size of the ON-phase zone (here 0.1) and the OFF-phase zone (0.3).
# It's a map of the excess events in phase, which are the pulsed events.

# In[ ]:


geom = WcsGeom.create(binsz=0.02 * u.deg, skydir=pos_target, width="5 deg")


#  Let's create an ON-map and an OFF-map:

# In[ ]:


on_map = Map.from_geom(geom)
off_map = Map.from_geom(geom)

events_vela_on = events_vela.select_parameter("PHASE", on_phase_range)
events_vela_off = events_vela.select_parameter("PHASE", off_phase_range)


# In[ ]:


fill_map_counts(on_map, events_vela_on)
fill_map_counts(off_map, events_vela_off)

# Defining alpha as the ratio of the ON and OFF phase zones
alpha = (on_phase_range[1] - on_phase_range[0]) / (
    off_phase_range[1] - off_phase_range[0]
)

# Create and fill excess map
# The pulsed events are the difference between the ON-phase count and alpha times the OFF-phase count
excess_map = on_map - off_map * alpha

# Plot excess map
excess_map.smooth(kernel="gauss", width=0.2 * u.deg).plot(add_cbar=True);


# ## Phase-resolved spectrum

# We can also do a phase-resolved spectrum. In order to do that, there is the class PhaseBackgroundEstimator. In a phase-resolved analysis, the background is estimated in the same sky region but in the OFF-phase zone.
# 
# We start by estimating the background with the class PhaseBackgroundEstimator. It takes the observations, the ON-region, and an ON- and OFF-phase zones (the same we defined for the phasogram and the phase-resolved map). It results in a gammapy.background.phase.PhaseBackgroundEstimator that serves as an input for other spectral analysis classes in Gammapy.

# In[ ]:


# The PhaseBackgroundEstimator uses the OFF-phase in the ON-region to estimate the background
bkg_estimator = PhaseBackgroundEstimator(
    observations=obs_list_vela,
    on_region=on_region,
    on_phase=on_phase_range,
    off_phase=off_phase_range,
)
bkg_estimator.run()
bkg_estimate = bkg_estimator.result


# The rest of the analysis is the same as for a standard spectral analysis with Gammapy. All the specificity of a phase-resolved analysis is contained in the PhaseBackgroundEstimator, where the background is estimated in the ON-region OFF-phase rather than in an OFF-region.
# 
# We can now extract a spectrum with the SpectrumExtraction class. It takes the reconstructed and the true energy binning. Both are expected to be a Quantity with unit energy, i.e. an array with an energy unit. EnergyBounds is a dedicated class to do it.

# In[ ]:


etrue = np.logspace(-2.5, 1, 100) * u.TeV
ereco = np.logspace(-2, 1, 30) * u.TeV

extraction = SpectrumExtraction(
    observations=obs_list_vela,
    bkg_estimate=bkg_estimate,
    containment_correction=True,
    e_true=etrue,
    e_reco=ereco,
)

extraction.run()
extraction.compute_energy_threshold(
    method_lo="energy_bias", bias_percent_lo=20
)


# Now let's a look at the files we just created with spectrum_observation.

# In[ ]:


extraction.spectrum_observations[0].peek()


# Now we'll fit a model to the spectrum with the `Fit` class. First we load a power law model with an initial value for the index and the amplitude and then wo do a likelihood fit. The fit results are printed below.

# In[ ]:


model = PowerLaw(
    index=4, amplitude="1.3e-9 cm-2 s-1 TeV-1", reference="0.02 TeV"
)

emin_fit, emax_fit = (0.04 * u.TeV, 0.4 * u.TeV)

for obs in extraction.spectrum_observations:
    obs.model = model
    obs.mask_fit = obs.counts.energy_mask(emin=emin_fit, emax=emax_fit)

joint_fit = Fit(extraction.spectrum_observations)
joint_result = joint_fit.run()

model.parameters.covariance = joint_result.parameters.covariance
print(joint_result)


# Now you might want to do the stacking here even if in our case there is only one observation which makes it superfluous.
# We can compute flux points by fitting the norm of the global model in energy bands.

# In[ ]:


e_edges = np.logspace(np.log10(0.04), np.log10(0.4), 7) * u.TeV

from gammapy.spectrum import SpectrumDatasetOnOffStacker

stacker = SpectrumDatasetOnOffStacker(extraction.spectrum_observations)
dataset = stacker.run()

dataset.model = model

fpe = FluxPointsEstimator(datasets=[dataset], e_edges=e_edges)

flux_points = fpe.run()
flux_points.table["is_ul"] = flux_points.table["ts"] < 1

amplitude_ref = 0.57 * 19.4e-14 * u.Unit("1 / (cm2 s MeV)")
spec_model_true = PowerLaw(
    index=4.5, amplitude=amplitude_ref, reference="20 GeV"
)

flux_points_dataset = FluxPointsDataset(data=flux_points, model=model)


# Now we can plot.

# In[ ]:


plt.figure(figsize=(8, 6))
ax_spectrum, ax_residual = flux_points_dataset.peek()

ax_spectrum.set_ylim([1e-14, 3e-11])
ax_residual.set_ylim([-1.7, 1.7])

spec_model_true.plot(
    ax=ax_spectrum,
    energy_range=(emin_fit, emax_fit),
    label="Reference model",
    c="black",
    linestyle="dashed",
    energy_power=2,
)

ax_spectrum.legend(loc="best")


# This tutorial suffers a bit from the lack of statistics: there were 9 Vela observations in the CTA DC1 while there is only one here. When done on the 9 observations, the spectral analysis is much better agreement between the input model and the gammapy fit.
