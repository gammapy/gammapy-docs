#!/usr/bin/env python
# coding: utf-8

# # Modeling and fitting 2D images using Gammapy
# 
# ## Prerequisites:
#  - To understand how a generel modelling and fiiting works in gammapy, please refer to the [analysis_3d tutorial](analysis_3d.ipynb)
# 
# ## Context:
# We often want the determine the position and morphology of an object. To do so, we don't necessarily have to resort to a full 3D fitting but can perform a simple image fitting, in particular, in an energy range where the PSF does not vary strongly, or if we want to explore a possible energy dependance of the morphology.
# 
# 
# ## Objective: 
# To localize a source and/or constrain its morphology.
# 
# ## Proposed approach:
# 
# The first step here, as in most analysis with DL3 data, is to create reduced datasets. For this, we will use the `Analysis` class to create a single set of stacked maps with a single bin in energy (thus, an *image* which behaves as a *cube*). This, we will then model with a spatial model of our choice, while keeping the spectral model fixed to an integrated power law.

# ## Setup
# As usual, we'll start with some general imports...

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from regions import CircleSkyRegion
from astropy.coordinates import Angle

import logging

log = logging.getLogger(__name__)


# Now let's import gammapy specific classes and functions

# In[ ]:


from gammapy.analysis import Analysis, AnalysisConfig


# ## Creating the config file

# Now, we create a config file for out analysis. You may load this from disc if you have a pre-defined config file.
# 
# Here, we use 3 simulated CTA runs of the galactic center.

# In[ ]:


config = AnalysisConfig()
# Selecting the observations
config.observations.datastore = "$GAMMAPY_DATA/cta-1dc/index/gps/"
config.observations.obs_ids = [110380, 111140, 111159]


# Technically, gammapy implements 2D analysis as a special case of 3D analysis (one one bin in energy). So, we must specify the type of analysis as *3D*, and define the geometry of the analysis.

# In[ ]:


config.datasets.type = "3d"
config.datasets.geom.wcs.skydir = {
    "lon": "0 deg",
    "lat": "0 deg",
    "frame": "galactic",
}  # The WCS geometry - centered on the galactic center
config.datasets.geom.wcs.fov = {"width": "10 deg", "height": "8 deg"}
config.datasets.geom.wcs.binsize = "0.02 deg"

# The FoV radius to use for cutouts
config.datasets.geom.selection.offset_max = 3.5 * u.deg

# We now fix the energy axis for the counts map - (the reconstructed energy binning)
config.datasets.geom.axes.energy.min = "0.1 TeV"
config.datasets.geom.axes.energy.max = "10 TeV"
config.datasets.geom.axes.energy.nbins = 1

config.datasets.geom.wcs.binsize_irf = 0.2 * u.deg


# In[ ]:


print(config)


# ## Getting the reduced dataset

# We now use the config file and create a single `MapDataset` containing `counts`, `background`, `exposure`, `psf` and `edisp` maps.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'analysis = Analysis(config)\nanalysis.get_observations()\nanalysis.get_datasets()')


# In[ ]:


print(analysis.datasets["stacked"])


# The counts and background maps have only one bin in reconstructed energy. The exposure and IRF maps are in true energy, and hence, have a different binning based upon the binning of the IRFs. We need not bother about them presently.

# In[ ]:


print(analysis.datasets["stacked"].counts)


# In[ ]:


print(analysis.datasets["stacked"].background_model.map)


# In[ ]:


print(analysis.datasets["stacked"].exposure)


# ## Modelling
# 
# Now, we define a model to be fitted to the dataset. **The important thing to note here is the dummy spectral model - an integrated powerlaw with only free normalisation**. Here, we use its YAML definition to load it:

# In[ ]:


model_config = """
components:
- name: GC-1
  type: SkyModel
  spatial:
    type: PointSpatialModel
    frame: galactic
    parameters:
    - name: lon_0
      value: 0.02
      unit: deg
    - name: lat_0 
      value: 0.01    
      unit: deg
  spectral:
    type: PowerLaw2SpectralModel
    parameters:
    - name: amplitude      
      value: 1.0e-12
      unit: cm-2 s-1 
    - name: index
      value: 2.0
      unit: ''
      frozen: true
    - name: emin
      value: 0.1
      unit: TeV
      frozen: true
    - name: emax
      value: 10.0
      unit: TeV
      frozen: true 
"""


# In[ ]:


analysis.set_models(model_config)


# We will freeze the parameters of the background

# In[ ]:


analysis.datasets["stacked"].background_model.parameters["norm"].frozen = True
analysis.datasets["stacked"].background_model.parameters["tilt"].frozen = True


# In[ ]:


# To run the fit
analysis.run_fit()


# In[ ]:


# To see the best fit values along with the errors
analysis.fit_result.parameters.to_table()


# In[ ]:




