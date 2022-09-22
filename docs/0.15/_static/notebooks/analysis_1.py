#!/usr/bin/env python
# coding: utf-8

# # First analysis
# 
# This notebook shows a simple example of a Crab analysis using the H.E.S.S. DL3 data release 1. It reduces the data to cube datasets and performs a simple 3D model fitting of the Crab nebula.
# 
# It uses the high level `Analysis` class to orchestrate data reduction. In its current state, `Analysis` supports the standard analysis cases of joint or stacked 3D and 1D analyses. It is instantiated with an `AnalysisConfig` object that gives access to analysis parameters either directly or via a YAML config file. 
# 
# To see what is happening under-the-hood and to get an idea of the internal API, a second notebook performs the same analysis without using the `Analysis` class. 
# 
# We will first show how to configure and run a stacked 3D analysis. The structure of the tutorial follows a typical analysis:
# 
# - Analysis configuration
# - Observation selection
# - Data reduction
# - Model fitting
# - Estimating flux points
# 
# Finally we will compare the results against a reference model.

# ## Setup

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


from pathlib import Path
from astropy import units as u
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling.models import create_crab_spectral_model


# ## Analysis configuration
# 
# For configuration of the analysis we use the [YAML](https://en.wikipedia.org/wiki/YAML) data format. YAML is a machine readable serialisation format, that is also friendly for humans to read. In this tutorial we will write the configuration file just using Python strings, but of course the file can be created and modified with any text editor of your choice.
# 
# Here is what the configuration for our analysis looks like:

# In[ ]:


config = AnalysisConfig()
# the AnalysisConfig gives access to the various parameters used from logging to reduced dataset geometries
print(config)


# ### Setting the data to use

# We want to use Crab runs from the H.E.S.S. DL3-DR1. We define here the datastore and a cone search of observations pointing with 5 degrees of the Crab nebula.
# Parameters can be set directly or as a python dict.

# In[ ]:


# We define the datastore containing the data
config.observations.datastore = "$GAMMAPY_DATA/hess-dl3-dr1"

# We define the cone search parameters
config.observations.obs_cone.frame = "icrs"
config.observations.obs_cone.lon = "83.633 deg"
config.observations.obs_cone.lat = "22.014 deg"
config.observations.obs_cone.radius = "5 deg"

# Equivalently we could have set parameters with a python dict
# config.observations.obs_cone = {"frame": "icrs", "lon": "83.633 deg", "lat": "22.014 deg", "radius": "5 deg"}


# ### Setting the reduced datasets geometry

# In[ ]:


# We want to perform a 3D analysis
config.datasets.type = "3d"
# We want to stack the data into a single reduced dataset
config.datasets.stack = True

# We fiw the WCS geometry of the datasets
config.datasets.geom.wcs.skydir = {
    "lon": "83.633 deg",
    "lat": "22.014 deg",
    "frame": "icrs",
}
config.datasets.geom.wcs.fov = {"width": "2 deg", "height": "2 deg"}
config.datasets.geom.wcs.binsize = "0.02 deg"

# We now fix the energy axis for the counts map
config.datasets.geom.axes.energy.min = "1 TeV"
config.datasets.geom.axes.energy.max = "10 TeV"
config.datasets.geom.axes.energy.nbins = 4

# We now fix the energy axis for the IRF maps (exposure, etc)
config.datasets.geom.axes.energy_true.min = "0.5 TeV"
config.datasets.geom.axes.energy_true.max = "20 TeV"
config.datasets.geom.axes.energy.nbins = 10


# ### Setting modeling and fitting parameters
# `Analysis` can perform a few modeling and fitting tasks besides data reduction. Parameters have then to be passed to the configuration object.

# In[ ]:


config.fit.fit_range.min = 1 * u.TeV
config.fit.fit_range.max = 10 * u.TeV
config.flux_points.energy = {"min": "1 TeV", "max": "10 TeV", "nbins": 3}


# We're all set. 
# But before we go on let's see how to save or import `AnalysisConfig` objects though YAML files.

# ### Using YAML configuration files
# 
# One can export/import the `AnalysisConfig` to/from a YAML file.

# In[ ]:


config.write("config.yaml", overwrite=True)


# In[ ]:


config = AnalysisConfig.read("config.yaml")
print(config)


# ## Running the analysis
# 
# We first create an `~gammapy.analysis.Analysis` object from our configuration.

# In[ ]:


analysis = Analysis(config)


# ###  Observation selection
# 
# We can directly select and load the observations from disk using `~gammapy.analysis.Analysis.get_observations()`:

# In[ ]:


analysis.get_observations()


# The observations are now available on the `Analysis` object. The selection corresponds to the following ids:

# In[ ]:


analysis.observations.ids


# To see how to explore observations, please refer to the following notebook: [CTA with Gammapy](cta.ipynb) or  [HESS with Gammapy](hess.ipynb) 

# ## Data reduction
# 
# Now we proceed to the data reduction. In the config file we have chosen a WCS map geometry, energy axis and decided to stack the maps. We can run the reduction using `.get_datasets()`:

# In[ ]:


get_ipython().run_cell_magic('time', '', 'analysis.get_datasets()')


# As we have chosen to stack the data, there is finally one dataset contained which we can print:

# In[ ]:


print(analysis.datasets["stacked"])


# As you can see the dataset comes with a predefined background model out of the data reduction, but no source model has been set yet.
# 
# The counts, exposure and background model maps are directly available on the dataset and can be printed and plotted:

# In[ ]:


counts = analysis.datasets["stacked"].counts
counts.smooth("0.05 deg").plot_interactive()


# ## Save dataset to disk
# 
# It is common to run the preparation step independent of the likelihood fit, because often the preparation of maps, PSF and energy dispersion is slow if you have a lot of data. We first create a folder:

# In[ ]:


path = Path("analysis_1")
path.mkdir(exist_ok=True)


# And then write the maps and IRFs to disk by calling the dedicated `write()` method:

# In[ ]:


filename = path / "crab-stacked-dataset.fits.gz"
analysis.datasets[0].write(filename, overwrite=True)


# ## Model fitting
# 
# Now we define a model to be fitted to the dataset. Here we use its YAML definition to load it:

# In[ ]:


model_config = """
components:
- name: crab
  type: SkyModel
  spatial:
    type: PointSpatialModel
    frame: icrs
    parameters:
    - name: lon_0
      value: 83.63
      unit: deg
    - name: lat_0 
      value: 22.14    
      unit: deg
  spectral:
    type: PowerLawSpectralModel
    parameters:
    - name: amplitude      
      value: 1.0e-12
      unit: cm-2 s-1 TeV-1
    - name: index
      value: 2.0
      unit: ''
    - name: reference
      value: 1.0
      unit: TeV
      frozen: true
"""


# Now we set the model on the analysis object:

# In[ ]:


analysis.set_models(model_config)


# Finally we run the fit:

# In[ ]:


analysis.run_fit()


# In[ ]:


print(analysis.fit_result)


# This is how we can write the model back to file again:

# In[ ]:


filename = path / "model-best-fit.yaml"
analysis.models.write(filename, overwrite=True)


# In[ ]:


get_ipython().system('cat analysis_1/model-best-fit.yaml')


# ### Flux points

# In[ ]:


analysis.get_flux_points(source="crab")


# In[ ]:


plt.figure(figsize=(8, 5))
ax_sed, ax_residuals = analysis.flux_points.peek()


# The flux points can be exported to a fits table following the format defined [here](https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/flux_points/index.html) 

# In[ ]:


filename = path / "flux-points.fits"
analysis.flux_points.write(filename, overwrite=True)


# ## What's next
# 
# You can look at the same analysis without the high level interface in [analysis_2](analysis_2.ipynb)
# 
# You can see how to perform a 1D spectral analysis of the same data in [spectrum analysis](spectrum_analysis.ipynb)

# In[ ]:




