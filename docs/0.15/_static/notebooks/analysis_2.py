#!/usr/bin/env python
# coding: utf-8

# # Second analysis
# 
# This notebook shows the same Crab analysis as performed in the [analysis_1 notebook](analysis_1.ipynb) but this time without the high level interface provided by the `Analysis` class. DL3 data release 1.
# 
# As before, we will reduce the data to cube datasets and perform a simple 3D model fitting of the Crab nebula.
# 
# The tutorial follows a typical analysis:
# 
# - Observation selection
# - Data reduction
# - Model fitting
# - Estimating flux points
# 
# but it gives more details on lower level API.
# 
# First, we setup the analysis by performing required imports.
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion


# In[ ]:


from gammapy.data import DataStore
from gammapy.maps import WcsGeom, MapAxis
from gammapy.cube import MapDatasetMaker, MapDataset, SafeMaskMaker
from gammapy.modeling.models import (
    SkyModel,
    PowerLawSpectralModel,
    PointSpatialModel,
)
from gammapy.modeling import Fit
from gammapy.spectrum import FluxPointsEstimator


# ## Defining the datastore and selecting observations
# 
# We first use the `~gammapy.data.DataStore` object to access the observations we want to analyse. Here the H.E.S.S. DL3 DR1. 

# In[ ]:


data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")


# We can now define an observation filter to select only the relevant observations. 
# Here we use a cone search which we define with a python dict.
# 
# We then filter the `ObservationTable` with `~gammapy.data.ObservationTable.select_observations()`.

# In[ ]:


selection = dict(
    type="sky_circle",
    frame="icrs",
    lon="83.633 deg",
    lat="22.014 deg",
    radius="5 deg",
)
selected_obs_table = data_store.obs_table.select_observations(selection)


# We can now retrieve the relevant observations by passing their `obs_id` to the`~gammapy.data.DataStore.get_observations()` method.

# In[ ]:


observations = data_store.get_observations(selected_obs_table["OBS_ID"])


# ## Preparing reduced datasets geometry
# 
# Now we define a reference geometry for our analysis, We choose a WCS based geometry with a binsize of 0.02 deg and also define an energy axis: 

# In[ ]:


energy_axis = MapAxis.from_edges(
    np.logspace(0.0, 1.0, 4), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(83.633, 22.014),
    binsz=0.02,
    width=(2, 2),
    coordsys="CEL",
    proj="CAR",
    axes=[energy_axis],
)

# Reduced IRFs are defined in true energy (i.e. not measured energy).
energy_axis_true = MapAxis.from_edges(
    np.logspace(-0.3, 1.3, 10), unit="TeV", name="energy", interp="log"
)


# Now we can define the target dataset with this geometry.

# In[ ]:


stacked = MapDataset.create(
    geom=geom, energy_axis_true=energy_axis_true, name="crab-stacked"
)


# ## Data reduction
# 
# ### Create the maker classes to be used
# 
# The `~gammapy.cube.MapDatasetMaker` object is initialized as well as the `~gammapy.cube.SafeMaskMaker` that carries here a maximum offset selection.

# In[ ]:


offset_max = 2.5 * u.deg
maker = MapDatasetMaker()
maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=offset_max)


# ### Perform the data reduction loop

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfor obs in observations:\n    # First a cutout of the target map is produced\n    cutout = stacked.cutout(obs.pointing_radec, width=2 * offset_max)\n    # A MapDataset is filled in this cutout geometry\n    dataset = maker.run(cutout, obs)\n    # The data quality cut is applied\n    dataset = maker_safe_mask.run(dataset, obs)\n    # The resulting dataset cutout is stacked onto the final one\n    stacked.stack(dataset)')


# ### Inspect the reduced dataset

# In[ ]:


stacked.counts.sum_over_axes().smooth(0.05 * u.deg).plot(
    stretch="sqrt", add_cbar=True
)


# ## Save dataset to disk
# 
# It is common to run the preparation step independent of the likelihood fit, because often the preparation of maps, PSF and energy dispersion is slow if you have a lot of data. We first create a folder:

# In[ ]:


path = Path("analysis_2")
path.mkdir(exist_ok=True)


# And then write the maps and IRFs to disk by calling the dedicated `~gammapy.cube.MapDataset.write()` method:

# In[ ]:


filename = path / "crab-stacked-dataset.fits.gz"
stacked.write(filename, overwrite=True)


# ## Define the model
# We first define the model, a `SkyModel`, as the combination of a point source `SpatialModel` with a powerlaw `SpectralModel`:

# In[ ]:


target_position = SkyCoord(ra=83.63308, dec=22.01450, unit="deg")
spatial_model = PointSpatialModel(
    lon_0=target_position.ra, lat_0=target_position.dec, frame="icrs"
)

spectral_model = PowerLawSpectralModel(
    index=2.702,
    amplitude=4.712e-11 * u.Unit("1 / (cm2 s TeV)"),
    reference=1 * u.TeV,
)

sky_model = SkyModel(
    spatial_model=spatial_model, spectral_model=spectral_model, name="crab"
)


# Now we assign this model to our reduced dataset:

# In[ ]:


stacked.models = sky_model


# ## Fit the model
# 
# The `~gammapy.modeling.Fit` class is orchestrating the fit, connecting the `stats` method of the dataset to the minimizer. By default, it uses `iminuit`.
# 
# Its contructor takes a list of dataset as argument.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit = Fit([stacked])\nresult = fit.run(optimize_opts={"print_level": 1})')


# The `FitResult` contains information on the fitted parameters.

# In[ ]:


result.parameters.to_table()


# ### Inspecting residuals
# 
# For any fit it is usefull to inspect the residual images. We have a few option on the dataset object to handle this. First we can use `.plot_residuals()` to plot a residual image, summed over all energies: 

# In[ ]:


stacked.plot_residuals(method="diff/sqrt(model)", vmin=-0.5, vmax=0.5)


# In addition we can aslo specify a region in the map to show the spectral residuals:

# In[ ]:


region = CircleSkyRegion(
    center=SkyCoord("83.63 deg", "22.14 deg"), radius=0.5 * u.deg
)


# In[ ]:


stacked.plot_residuals(
    region=region, method="diff/sqrt(model)", vmin=-0.5, vmax=0.5
)


# We can also directly access the `.residuals()` to get a map, that we can plot interactively:

# In[ ]:


residuals = stacked.residuals(method="diff")
residuals.smooth("0.08 deg").plot_interactive(
    cmap="coolwarm", vmin=-0.1, vmax=0.1, stretch="linear", add_cbar=True
)


# ### Inspecting fit statistic profiles
# 
# To check the quality of the fit it is also useful to plot fit statistic profiles for specific parameters.
# For this we use `~gammapy.modeling.Fit.stat_profile()`.

# In[ ]:


profile = fit.stat_profile(parameter="lon_0")


# For a good fit and error estimate the profile should be parabolic, if we plot it:

# In[ ]:


total_stat = result.total_stat
plt.plot(profile["values"], profile["stat"] - total_stat)
plt.xlabel("Lon (deg)")
plt.ylabel("Delta TS")


# ## Plot the fitted spectrum

# ### Making a butterfly plot 
# 
# The `SpectralModel` component can be used to produce a, so-called, butterfly plot showing the enveloppe of the model taking into account parameter uncertainties.
# 
# To do so, we have to copy the part of the covariance matrix stored on the `FitResult` on the model parameters:

# In[ ]:


spec = sky_model.spectral_model

# set covariance on the spectral model
covar = result.parameters.get_subcovariance(spec.parameters)
spec.parameters.covariance = covar


# Now we can actually do the plot using the `plot_error` method:

# In[ ]:


energy_range = [1, 10] * u.TeV
spec.plot(energy_range=energy_range, energy_power=2)
ax = spec.plot_error(energy_range=energy_range, energy_power=2)


# ### Computing flux points
# 
# We can now compute some flux points using the `~gammapy.spectrum.FluxPointsEstimator`. 
# 
# Besides the list of datasets to use, we must provide it the energy intervals on which to compute flux points as well as the model component name. 

# In[ ]:


e_edges = [1, 2, 4, 10] * u.TeV
fpe = FluxPointsEstimator(datasets=[stacked], e_edges=e_edges, source="crab")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'flux_points = fpe.run()')


# In[ ]:


ax = spec.plot_error(energy_range=energy_range, energy_power=2)
flux_points.plot(ax=ax, energy_power=2)

