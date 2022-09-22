#!/usr/bin/env python
# coding: utf-8

# # Spectrum simulation
# 
# ## Prerequisites
# 
# - Knowledge of spectral extraction and datasets used in gammapy, see for instance the [spectral analysis tutorial](spectrum_analysis.ipynb)
# 
# ## Context
# 
# To simulate a specific observation, it is not always necessary to simulate the full photon list. For many uses cases, simulating directly a reduced binned dataset is enough: the IRFs reduced in the correct geometry are combined with a source model to predict an actual number of counts per bin. The latter is then used to simulate a reduced dataset using Poisson probability distribution.
# 
# This can be done to check the feasibility of a measurement, to test whether fitted parameters really provide a good fit to the data etc.
# 
# Here we will see how to perform a 1D spectral simulation of a CTA observation, in particular, we will generate OFF observations following the template background stored in the CTA IRFs.
# 
# **Objective: simulate a number of spectral ON-OFF observations of a source with a power-law spectral model with CTA using the CTA 1DC response, fit them with the assumed spectral model and check that the distribution of fitted parameters is consistent with the input values.**
# 
# ## Proposed approach:
# 
# We will use the following classes:
# 
# * `~gammapy.spectrum.SpectrumDatasetOnOff`
# * `~gammapy.spectrum.SpectrumDataset`
# * `~gammapy.irf.load_cta_irfs`
# * `~gammapy.modeling.models.PowerLawSpectralModel`

# ## Setup
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from gammapy.spectrum import (
    SpectrumDatasetOnOff,
    SpectrumDataset,
    SpectrumDatasetMaker,
)
from gammapy.modeling import Fit, Parameter
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SpectralModel,
    SkyModel,
)
from gammapy.irf import load_cta_irfs
from gammapy.data import Observation
from gammapy.maps import MapAxis


# ## Simulation of a single spectrum
# 
# To do a simulation, we need to define the observational parameters like the livetime, the offset, the assumed integration radius, the energy range to perform the simulation for and the choice of spectral model. We then use an in-memory observation which is convolved with the IRFs to get the predicted number of counts. This is Poission fluctuated using the `fake()` to get the simulated counts for each observation.  

# In[ ]:


# Define simulation parameters parameters
livetime = 1 * u.h
pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
offset = 0.5 * u.deg
# Reconstructed and true energy axis
energy_axis = MapAxis.from_edges(
    np.logspace(-0.5, 1.0, 10), unit="TeV", name="energy", interp="log"
)
energy_axis_true = MapAxis.from_edges(
    np.logspace(-1.2, 2.0, 31), unit="TeV", name="energy", interp="log"
)

on_region_radius = Angle("0.11 deg")
on_region = CircleSkyRegion(center=pointing, radius=on_region_radius)


# In[ ]:


# Define spectral model - a simple Power Law in this case
model_simu = PowerLawSpectralModel(
    index=3.0,
    amplitude=2.5e-12 * u.Unit("cm-2 s-1 TeV-1"),
    reference=1 * u.TeV,
)
print(model_simu)
# we set the sky model used in the dataset
model = SkyModel(spectral_model=model_simu)


# In[ ]:


# Load the IRFs
# In this simulation, we use the CTA-1DC irfs shipped with gammapy.
irfs = load_cta_irfs(
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)


# In[ ]:


obs = Observation.create(pointing=pointing, livetime=livetime, irfs=irfs)
print(obs)


# In[ ]:


# Make the SpectrumDataset
dataset_empty = SpectrumDataset.create(
    e_reco=energy_axis.edges, e_true=energy_axis_true.edges, region=on_region
)
maker = SpectrumDatasetMaker(selection=["aeff", "edisp", "background"])
dataset = maker.run(dataset_empty, obs)


# In[ ]:


# Set the model on the dataset, and fake
dataset.model = model
dataset.fake(random_state=42)
print(dataset)


# You can see that backgound counts are now simulated

# ### OnOff analysis
# 
# To do `OnOff` spectral analysis, which is the usual science case, the standard would be to use `SpectrumDatasetOnOff`, which uses the acceptance to fake off-counts 

# In[ ]:


dataset_onoff = SpectrumDatasetOnOff(
    aeff=dataset.aeff,
    edisp=dataset.edisp,
    models=model,
    livetime=livetime,
    acceptance=1,
    acceptance_off=5,
)
dataset_onoff.fake(background_model=dataset.background)
print(dataset_onoff)


# You can see that off counts are now simulated as well. We now simulate several spectra using the same set of observation conditions.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nn_obs = 100\ndatasets = []\n\nfor idx in range(n_obs):\n    dataset_onoff.fake(\n        random_state=idx,\n        background_model=dataset.background,\n        name=f"obs_{idx}",\n    )\n    datasets.append(dataset_onoff.copy())')


# Before moving on to the fit let's have a look at the simulated observations.

# In[ ]:


n_on = [dataset.counts.data.sum() for dataset in datasets]
n_off = [dataset.counts_off.data.sum() for dataset in datasets]
excess = [dataset.excess.data.sum() for dataset in datasets]

fix, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].hist(n_on)
axes[0].set_xlabel("n_on")
axes[1].hist(n_off)
axes[1].set_xlabel("n_off")
axes[2].hist(excess)
axes[2].set_xlabel("excess");


# Now, we fit each simulated spectrum individually 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'results = []\nfor dataset in datasets:\n    dataset.models = model.copy()\n    fit = Fit([dataset])\n    result = fit.optimize()\n    results.append(\n        {\n            "index": result.parameters["index"].value,\n            "amplitude": result.parameters["amplitude"].value,\n        }\n    )')


# We take a look at the distribution of the fitted indices. This matches very well with the spectrum that we initially injected, index=2.1

# In[ ]:


index = np.array([_["index"] for _ in results])
plt.hist(index, bins=10, alpha=0.5)
plt.axvline(x=model_simu.parameters["index"].value, color="red")
print(f"index: {index.mean()} += {index.std()}")


# ## Exercises
# 
# * Change the observation time to something longer or shorter. Does the observation and spectrum results change as you expected?
# * Change the spectral model, e.g. add a cutoff at 5 TeV, or put a steep-spectrum source with spectral index of 4.0
# * Simulate spectra with the spectral model we just defined. How much observation duration do you need to get back the injected parameters?

# In[ ]:




