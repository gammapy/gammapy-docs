#!/usr/bin/env python
# coding: utf-8

# # 3D simulation and fitting
# 
# This tutorial shows how to do a 3D map-based simulation and fit.
# 
# For a tutorial on how to do a 3D map analyse of existing data, see the [analysis_3d](analysis_3d.ipynb) tutorial.
# 
# This can be useful to do a performance / sensitivity study, or to evaluate the capabilities of Gammapy or a given analysis method. Note that is is a binned simulation as is e.g. done also in Sherpa for Chandra, not an event sampling and anbinned analysis as is done e.g. in the Fermi ST or ctools.

# ## Imports and versions

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.irf import load_cta_irfs
from gammapy.maps import WcsGeom, MapAxis
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.modeling.models import GaussianSpatialModel
from gammapy.modeling.models import SkyModel
from gammapy.cube import MapDataset, MapDatasetMaker, SafeMaskMaker
from gammapy.modeling import Fit
from gammapy.data import Observation


# In[ ]:


get_ipython().system('gammapy info --no-envvar --no-dependencies --no-system')


# ## Simulation

# We will simulate using the CTA-1DC IRFs shipped with gammapy. Note that for dedictaed CTA simulations, you can simply use [`Observation.from_caldb()`]() without having to externally load the IRFs

# In[ ]:


# Loading IRFs
irfs = load_cta_irfs(
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)


# In[ ]:


# Define the observation parameters (typically the observation duration and the pointing position):
livetime = 2.0 * u.hr
pointing = SkyCoord(0, 0, unit="deg", frame="galactic")


# In[ ]:


# Define map geometry for binned simulation
energy_reco = MapAxis.from_edges(
    np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(0, 0), binsz=0.02, width=(6, 6), coordsys="GAL", axes=[energy_reco]
)
# It is usually useful to have a separate binning for the true energy axis
energy_true = MapAxis.from_edges(
    np.logspace(-1.5, 1.5, 30), unit="TeV", name="energy", interp="log"
)


# In[ ]:


# Define sky model to used simulate the data.
# Here we use a Gaussian spatial model and a Power Law spectral model.
spatial_model = GaussianSpatialModel(
    lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.3 deg", frame="galactic"
)
spectral_model = PowerLawSpectralModel(
    index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)
model_simu = SkyModel(
    spatial_model=spatial_model,
    spectral_model=spectral_model,
    name="model_simu",
)
print(model_simu)


# Now, comes the main part of dataset simulation. We create an in-memory observation and an empty dataset. We then predict the number of counts for the given model, and Poission fluctuate it using `fake()` to make a simulated counts maps. Keep in mind that it is important to specify the `selection` of the maps that you want to produce 

# In[ ]:


# Create an in-memory observation
obs = Observation.create(pointing=pointing, livetime=livetime, irfs=irfs)
print(obs)
# Make the MapDataset
empty = MapDataset.create(geom)
maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg)
dataset = maker.run(empty, obs)
dataset = maker_safe_mask.run(dataset, obs)
print(dataset)


# In[ ]:


# Add the model on the dataset and Poission fluctuate
dataset.models = model_simu
dataset.fake()
# Do a print on the dataset - there is now a counts maps
print(dataset)


# Now use this dataset as you would in all standard analysis. You can plot the maps, or proceed with your custom analysis. 
# In the next section, we show the standard 3D fitting as in [analysis_3d](analysis_3d.ipynb).

# In[ ]:


# To plot, eg, counts:
dataset.counts.smooth(0.05 * u.deg).plot_interactive(
    add_cbar=True, stretch="linear"
)


# ## Fit
# 
# In this section, we do a usual 3D fit with the same model used to simulated the data and see the stability of the simulations. Often, it is useful to simulate many such datasets and look at the distribution of the reconstructed parameters.

# In[ ]:


# Make a copy of the dataset
dataset1 = dataset.copy()


# In[ ]:


# Define sky model to fit the data
spatial_model1 = GaussianSpatialModel(
    lon_0="0.1 deg", lat_0="0.1 deg", sigma="0.5 deg", frame="galactic"
)
spectral_model1 = PowerLawSpectralModel(
    index=2, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)
model_fit = SkyModel(
    spatial_model=spatial_model1,
    spectral_model=spectral_model1,
    name="model_fit",
)

dataset1.models = model_fit
print(model_fit)


# In[ ]:


# We do not want to fit the background in this case, so we will freeze the parameters
background_model = dataset1.background_model
background_model.parameters["norm"].value = 1.0
background_model.parameters["norm"].frozen = True
background_model.parameters["tilt"].frozen = True

print(background_model)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit = Fit([dataset1])\nresult = fit.run(optimize_opts={"print_level": 1})')


# In[ ]:


dataset1.plot_residuals(method="diff/sqrt(model)", vmin=-0.5, vmax=0.5)


# Compare the injected and fitted models: 

# In[ ]:


print("True model: \n", model_simu, "\n\n Fitted model: \n", model_fit)


# Get the errors on the fitted parameters from the parameter table

# In[ ]:


result.parameters.to_table()

