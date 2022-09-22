#!/usr/bin/env python
# coding: utf-8

# # Simulating and fitting a time varying source
# 
# ## Prerequisites:
# 
# - To understand how a single binned simulation works, please refer to [spectrum_simulation](../1D/spectrum_simulation.ipynb) [simulate_3d](../3D/simulate_3d.ipynb) for 1D and 3D simulations respectively.
# - For details of light curve extraction using gammapy, refer to the two tutorials [light_curve](light_curve.ipynb) and [light_curve_flare](light_curve_flare.ipynb) 
# 
# ## Context
# 
# Frequently, studies of variable sources (eg: decaying GRB light curves, AGN flares, etc) require time variable simulations. For most use cases, generating an event list is an overkill, and it suffices to use binned simulations using a temporal model.
# 
# **Objective: Simulate and fit a time decaying light curve of a source with CTA using the CTA 1DC response**
# 
# ## Proposed approach:
# 
# We will simulate 10 spectral datasets within given time intervals (Good Time Intervals) following a given spectral (a power law) and temporal profile (an exponential decay, with a decay time of 6 hr ). These are then analysed using the light curve estimator to obtain flux points. Then, we re-fit the simulated datasets to reconstruct back the injected profiles.
# 
# In summary, necessary steps are:
# 
# - Choose observation parameters including a list of `gammapy.data.GTI`
# - Define temporal and spectral models from :ref:model-gallery as per science case
# - Perform the simulation (in 1D or 3D)
# - Extract the light curve from the reduced dataset as shown in [light curve notebook](light_curve.ipynb)
# - Optionally, we show here how to fit the simulated datasets using a source model 
# 
# 
# ## Setup 
# 
# As usual, we'll start with some general imports...

# ## Setup

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import logging

log = logging.getLogger(__name__)


# And some gammapy specific imports

# In[ ]:


from gammapy.data import Observation
from gammapy.irf import load_cta_irfs
from gammapy.datasets import SpectrumDataset, Datasets
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    ExpDecayTemporalModel,
    SkyModel,
)
from gammapy.maps import MapAxis, RegionGeom
from gammapy.estimators import LightCurveEstimator
from gammapy.makers import SpectrumDatasetMaker
from gammapy.modeling import Fit


# ## Simulating a light curve
# 
# We will simulate 10 datasets using an `PowerLawSpectralModel` and a `ExpDecayTemporalModel`. The important thing to note here is how to attach a different `GTI` to each dataset.

# In[ ]:


# Loading IRFs
irfs = load_cta_irfs(
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)


# In[ ]:


# Reconstructed and true energy axis
energy_axis = MapAxis.from_edges(
    np.logspace(-0.5, 1.0, 10), unit="TeV", name="energy", interp="log"
)
energy_axis_true = MapAxis.from_edges(
    np.logspace(-1.2, 2.0, 31), unit="TeV", name="energy_true", interp="log"
)

geom = RegionGeom.create("galactic;circle(0, 0, 0.11)", axes=[energy_axis])


# In[ ]:


# Pointing position
pointing = SkyCoord(0.5, 0.5, unit="deg", frame="galactic")


# Note that observations are usually conducted in  Wobble mode, in which the source is not in the center of the camera. This allows to have a symmetrical sky position from which background can be estimated.

# In[ ]:


# Define the source model: A combination of spectral and temporal model

gti_t0 = Time("2020-03-01")
spectral_model = PowerLawSpectralModel(
    index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)
temporal_model = ExpDecayTemporalModel(t0="6 h", t_ref=gti_t0.mjd * u.d)

model_simu = SkyModel(
    spectral_model=spectral_model,
    temporal_model=temporal_model,
    name="model-simu",
)


# In[ ]:


# Look at the model
model_simu.parameters.to_table()


# Now, define the start and observation livetime wrt to the reference time, `gti_t0`

# In[ ]:


n_obs = 10
tstart = [1, 2, 3, 5, 8, 10, 20, 22, 23, 24] * u.h
lvtm = [55, 25, 26, 40, 40, 50, 40, 52, 43, 47] * u.min


# Now perform the simulations

# In[ ]:


datasets = Datasets()

empty = SpectrumDataset.create(
    geom=geom, energy_axis_true=energy_axis_true, name="empty"
)

maker = SpectrumDatasetMaker(selection=["exposure", "background", "edisp"])

for idx in range(n_obs):
    obs = Observation.create(
        pointing=pointing,
        livetime=lvtm[idx],
        tstart=tstart[idx],
        irfs=irfs,
        reference_time=gti_t0,
        obs_id=idx,
    )
    empty_i = empty.copy(name=f"dataset-{idx}")
    dataset = maker.run(empty_i, obs)
    dataset.models = model_simu
    dataset.fake()
    datasets.append(dataset)


# The reduced datasets have been successfully simulated. Let's take a quick look into our datasets.

# In[ ]:


datasets.info_table()


# ## Extract the lightcurve
# 
# This section uses standard light curve estimation tools for a 1D extraction. Only a spectral model needs to be defined in this case. Since the estimator returns the integrated flux separately for each time bin, the temporal model need not be accounted for at this stage.

# In[ ]:


# Define the model:
spectral_model = PowerLawSpectralModel(
    index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)
model_fit = SkyModel(spectral_model=spectral_model, name="model-fit")


# In[ ]:


# Attach model to all datasets
datasets.models = model_fit


# In[ ]:


get_ipython().run_cell_magic('time', '', 'lc_maker_1d = LightCurveEstimator(\n    energy_edges=[0.3, 10] * u.TeV,\n    source="model-fit",\n    selection_optional=["ul"],\n)\nlc_1d = lc_maker_1d.run(datasets)')


# In[ ]:


ax = lc_1d.plot(marker="o", label="3D")


# We have the reconstructed lightcurve at this point. Further standard analysis might involve modeling the temporal profiles with an analytical or theoretical model. You may do this using your favourite fitting package, one possible option being `curve_fit` inside `scipy.optimize`.
# 
# In the next section, we show how to simultaneously fit the all datasets using a given temporal model. This does a joint fitting across the different datasets, while simultaneously minimising across the temporal model parameters as well. We will fit the amplitude, spectral index and the decay time scale. Note that `t_ref` should be fixed by default for the `ExpDecayTemporalModel`. 
# 
# For modelling and fitting more complex flares, you should attach the relevant model to each group of `datasets`. The parameters of a model in a given group of dataset will be tied. For more details on joint fitting in gammapy, see [here](../2D/modeling_2D.ipynb).

# ## Fit the datasets

# In[ ]:


# Define the model:
spectral_model1 = PowerLawSpectralModel(
    index=2.0, amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV"
)
temporal_model1 = ExpDecayTemporalModel(t0="10 h", t_ref=gti_t0.mjd * u.d)

model = SkyModel(
    spectral_model=spectral_model1,
    temporal_model=temporal_model1,
    name="model-test",
)


# In[ ]:


model.parameters.to_table()


# In[ ]:


datasets.models = model


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Do a joint fit\nfit = Fit()\nresult = fit.run(datasets=datasets)')


# In[ ]:


result.parameters.to_table()


# We see that the fitted parameters match well with the simulated ones!

# ## Exercises
# 
# 1. Re-do the analysis with `MapDataset` instead of `SpectralDataset`
# 2. Model the flare of PKS 2155-304 which you obtained using the [light curve flare tutorial](light_curve_flare.ipynb). Use a combination of a Gaussian and Exponential flare profiles, and fit using `scipy.optimize.curve_fit`
# 3. Do a joint fitting of the datasets.

# In[ ]:




