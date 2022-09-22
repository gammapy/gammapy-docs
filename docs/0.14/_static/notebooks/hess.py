#!/usr/bin/env python
# coding: utf-8

# # Analysis of H.E.S.S. DL3 data with Gammapy
# 
# In September 2018 the [H.E.S.S.](https://www.mpi-hd.mpg.de/hfm/HESS) collaboration released a small subset of archival data in FITS format. This tutorial explains how to analyse this data with Gammapy. We will analyse four observation runs of the Crab nebula, which are part of the [H.E.S.S. first public test data release](https://www.mpi-hd.mpg.de/hfm/HESS/pages/dl3-dr1/). The data was release without corresponding background models. In [background_model.ipynb](background_model.ipynb) we show how to make a simple background model, which is also used in this tutorial. The background model is not perfect; it assumes radial symmetry and is in general derived from only a few observations, but still good enough for a reliable analysis > 1TeV.
# 
# **Note:** The high level `Analysis` class is a new feature added in Gammapy v0.14. In the curret state it supports the standard analysis cases of a joint or stacked 3D and 1D analysis. It provides only limited access to analaysis parameters via the config file. It is expected that the format of the YAML config will be extended and change in future Gammapy versions.
# 
# We will first show how to configure and run a stacked 3D analysis and then address the classical spectral analysis using reflected regions later. The structure of the tutorial follows a typical analysis:
# 
# - Analysis configuration
# - Observation slection
# - Data reduction
# - Model fitting
# - Estimating flux points
# 
# Finally we will compare the results against a reference model.

# ## Setup

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


import yaml
from pathlib import Path
from regions import CircleSkyRegion
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.scripts import Analysis, AnalysisConfig
from gammapy.modeling.models import create_crab_spectral_model


# ## Analysis configuration
# 
# For configuration of the analysis we use the [YAML](https://en.wikipedia.org/wiki/YAML) data format. YAML is a machine readable serialisation format, that is also friendly for humans to read. In this tutorial we will write the configuration file just using Python strings, but of course the file can be created and modified with any text editor of your choice.
# 
# Here is what the configuration for our analysis looks like:

# In[ ]:


config_str = """
general:
    logging:
        level: INFO
    outdir: .

observations:
    datastore: $GAMMAPY_DATA/hess-dl3-dr1/hess-dl3-dr3-with-background.fits.gz
    filters:
        - filter_type: par_value
          value_param: Crab
          variable: TARGET_NAME

datasets:
    dataset-type: MapDataset
    stack-datasets: true
    offset-max: 2.5 deg
    geom:
        skydir: [83.633, 22.014]
        width: [5, 5]
        binsz: 0.02
        coordsys: CEL
        proj: TAN
        axes:
          - name: energy
            hi_bnd: 10
            lo_bnd: 1
            nbin: 5
            interp: log
            node_type: edges
            unit: TeV

fit:
    fit_range:
        max: 30 TeV
        min: 1 TeV

flux-points:
    fp_binning:
        lo_bnd: 1
        hi_bnd: 10
        interp: log
        nbin: 3
        unit: TeV
"""


# We first create an `AnalysiConfig` object from it:

# In[ ]:


config = AnalysisConfig(config_str)


# ##  Observation selection
# 
# Now we create the high level `Analysis` object from the config object:

# In[ ]:


analysis = Analysis(config)


# And directly select and load the observatiosn from disk using `.get_observations()`:

# In[ ]:


analysis.get_observations()


# The observations are now availabe on the `Analysis` object. The selection corresponds to the following ids:

# In[ ]:


analysis.observations.ids


# Now we can access and inspect individual observations by accessing with the observation id:

# In[ ]:


print(analysis.observations["23592"])


# And also show a few overview plots using the `.peek()` method:

# In[ ]:


analysis.observations["23592"].peek()


# ## Data reduction
# 
# Now we proceed to the data reduction. In the config file we have chosen a WCS map geometry, energy axis and decided to stack the maps. We can run the reduction using `.get_datasets()`:

# In[ ]:


get_ipython().run_cell_magic('time', '', 'analysis.get_datasets()')


# As we have chosen to stack the data, there is finally one dataset contained:

# In[ ]:


analysis.datasets.names


# We can print the dataset as well:

# In[ ]:


print(analysis.datasets["stacked"])


# As you can see the dataset comes with a predefined background model out of the data reduction, but no source model has been set yet.
# 
# The counts, exposure and background model maps are directly available on the dataset and can be printed and plotted:

# In[ ]:


counts = analysis.datasets["stacked"].counts


# In[ ]:


print(counts)


# In[ ]:


counts.smooth("0.05 deg").plot_interactive()


# ## Model fitting
# 
# Now we define a model to be fitted to the dataset:

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


analysis.set_model(model_config)


# In[ ]:


print(analysis.model)


# In[ ]:


print(analysis.model["crab"])


# Finally we run the fit:

# In[ ]:


analysis.run_fit()


# In[ ]:


print(analysis.fit_result)


# This is how we can write the model back to file again:

# In[ ]:


analysis.model.to_yaml("model-best-fit.yaml")


# In[ ]:


get_ipython().system('cat model-best-fit.yaml')


# ### Inspecting residuals
# 
# For any fit it is usefull to inspect the residual images. We have a few option on the dataset object to handle this. First we can use `.plot_residuals()` to plot a residual image, summed over all energies: 

# In[ ]:


analysis.datasets["stacked"].plot_residuals(
    method="diff/sqrt(model)", vmin=-0.5, vmax=0.5
);


# In addition we can aslo specify a region in the map to show the spectral residuals:

# In[ ]:


region = CircleSkyRegion(
    center=SkyCoord("83.63 deg", "22.14 deg"), radius=0.5 * u.deg
)


# In[ ]:


analysis.datasets["stacked"].plot_residuals(
    region=region, method="diff/sqrt(model)", vmin=-0.5, vmax=0.5
);


# We can also directly access the `.residuals()` to get a map, that we can plot interactively:

# In[ ]:


residuals = analysis.datasets["stacked"].residuals(method="diff")
residuals.smooth("0.08 deg").plot_interactive(
    cmap="coolwarm", vmin=-0.1, vmax=0.1, stretch="linear", add_cbar=True
)


# ### Inspecting likelihood profiles
# 
# To check the quality of the fit it is also useful to plot likelihood profiles for specific parameters. For this we use `analysis.fit.likelihood_profile()`

# In[ ]:


profile = analysis.fit.likelihood_profile(parameter="lon_0")


# For a good fit and error estimate the profile should be parabolic, if we plot it:

# In[ ]:


total_stat = analysis.fit_result.total_stat
plt.plot(profile["values"], profile["likelihood"] - total_stat)
plt.xlabel("Lon (deg)")
plt.ylabel("Delta TS")


# ### Flux points

# In[ ]:


analysis.get_flux_points(source="crab")


# In[ ]:


plt.figure(figsize=(8, 5))
ax_sed, ax_residuals = analysis.flux_points.peek()
crab_spectrum = create_crab_spectral_model("hess_pl")
crab_spectrum.plot(
    ax=ax_sed,
    energy_range=[1, 10] * u.TeV,
    energy_power=2,
    flux_unit="erg-1 cm-2 s-1",
)


# ## Exercises
# 
# - Run a spectral analysis using reflected regions without stacking the datasets. You can use `AnalysisConfig.from_template("1d")` to get an example configuration file. Add the resulting flux points to the SED plotted above. 
# 
