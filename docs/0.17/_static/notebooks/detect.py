#!/usr/bin/env python
# coding: utf-8

# # Source detection with Gammapy
# 
# ## Context
# 
# The first task in a source catalogue production is to identify significant excesses in the data that can be associated to unknown sources and provide a preliminary parametrization in term of position, extent, and flux. In this notebook we will use Fermi-LAT data to illustrate how to detect candidate sources in counts images with known background.
# 
# **Objective: build a list of significant excesses in a Fermi-LAT map**
# 
# 
# ## Proposed approach 
# 
# This notebook show how to do source detection with Gammapy using the methods available in `~gammapy.estimators`.
# We will use images from a Fermi-LAT 3FHL high-energy Galactic center dataset to do this:
# 
# * perform adaptive smoothing on counts image
# * produce 2-dimensional test-statistics (TS)
# * run a peak finder to detect point-source candidates
# * compute Li & Ma significance images
# * estimate source candidates radius and excess counts
# 
# Note that what we do here is a quick-look analysis, the production of real source catalogs use more elaborate procedures.
# 
# We will work with the following functions and classes:
# 
# * `~gammapy.maps.WcsNDMap`
# * `~gammapy.estimators.ASmoothEstimator`
# * `~gammapy.estimators.TSMapEstimator`
# * `~gammapy.estimators.utils.find_peaks`

# ## Setup
# 
# As always, let's get started with some setup ...

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gammapy.maps import Map
from gammapy.estimators import (
    ASmoothMapEstimator,
    TSMapEstimator,
)
from gammapy.estimators.utils import find_peaks
from gammapy.datasets import MapDataset
from gammapy.catalog import SOURCE_CATALOGS
from gammapy.modeling.models import (
    BackgroundModel,
    SkyModel,
    PowerLawSpectralModel,
    PointSpatialModel,
)
from gammapy.irf import PSFMap, EnergyDependentTablePSF
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


# ## Read in input images
# 
# We first read in the counts cube and sum over the energy axis:

# In[ ]:


counts = Map.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts-cube.fits.gz"
)
background = Map.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-background-cube.fits.gz"
)
background = BackgroundModel(background, datasets_names=["fermi-3fhl-gc"])

exposure = Map.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-exposure-cube.fits.gz"
)
# unit is not properly stored on the file. We add it manually
exposure.unit = "cm2s"
mask_safe = counts.copy(data=np.ones_like(counts.data).astype("bool"))

psf = EnergyDependentTablePSF.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-psf-cube.fits.gz"
)
psfmap = PSFMap.from_energy_dependent_table_psf(psf)

dataset = MapDataset(
    counts=counts,
    models=[background],
    exposure=exposure,
    psf=psfmap,
    mask_safe=mask_safe,
    name="fermi-3fhl-gc",
)

dataset = dataset.to_image()


# ## Adaptive smoothing
#  
# For visualisation purpose it can be nice to look at a smoothed counts image. This can be performed using the adaptive smoothing algorithm from [Ebeling et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.368...65E/abstract).
#      
# In the following example the `threshold` argument gives the minimum significance expected, values below are clipped.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'scales = u.Quantity(np.arange(0.05, 1, 0.05), unit="deg")\nsmooth = ASmoothMapEstimator(threshold=3, scales=scales)\nimages = smooth.run(dataset)\n\nplt.figure(figsize=(15, 5))\nimages["counts"].plot(add_cbar=True, vmax=10)')


# ## TS map estimation
# 
# The Test Statistic, TS = 2 ∆ log L ([Mattox et al. 1996](https://ui.adsabs.harvard.edu/abs/1996ApJ...461..396M/abstract)), compares the likelihood function L optimized with and without a given source.
# The TS map is computed by fitting by a single amplitude parameter on each pixel as described in Appendix A of [Stewart (2009)](https://ui.adsabs.harvard.edu/abs/2009A%26A...495..989S/abstract). The fit is simplified by finding roots of the derivative of the fit statistics (default settings use [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method)).
# 
# We first need to define the model that will be used to test for the existence of a source. Here, we use a point source.

# In[ ]:


spatial_model = PointSpatialModel()
spectral_model = PowerLawSpectralModel(index=2)
model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'estimator = TSMapEstimator(model)\nimages = estimator.run(dataset)')


# ### Plot resulting images

# In[ ]:


plt.figure(figsize=(15, 5))
images["sqrt_ts"].plot(add_cbar=True);


# In[ ]:


plt.figure(figsize=(15, 5))
images["flux"].plot(add_cbar=True, stretch="sqrt", vmin=0);


# In[ ]:


plt.figure(figsize=(15, 5))
images["niter"].plot(add_cbar=True);


# ## Source candidates
# 
# Let's run a peak finder on the `sqrt_ts` image to get a list of point-sources candidates (positions and peak `sqrt_ts` values).
# The `find_peaks` function performs a local maximun search in a sliding window, the argument `min_distance` is the minimum pixel distance between peaks (smallest possible value and default is 1 pixel).

# In[ ]:


sources = find_peaks(images["sqrt_ts"], threshold=8, min_distance=1)
nsou = len(sources)
sources


# In[ ]:


# Plot sources on top of significance sky image
plt.figure(figsize=(15, 5))

_, ax, _ = images["sqrt_ts"].plot(add_cbar=True)

ax.scatter(
    sources["ra"],
    sources["dec"],
    transform=plt.gca().get_transform("icrs"),
    color="none",
    edgecolor="w",
    marker="o",
    s=600,
    lw=1.5,
);


# Note that we used the instrument point-spread-function (PSF) as kernel, so the hypothesis we test is the presence of a point source. In order to test for extended sources we would have to use as kernel an extended template convolved by the PSF. Alternatively, we can compute the significance of an extended excess using the Li & Ma formalism, which is faster as no fitting is involve.

# ## What next?
# 
# In this notebook, we have seen how to work with images and compute TS and significance images from counts data, if a background estimate is already available.
# 
# Here's some suggestions what to do next:
# 
# - Look how background estimation is performed for IACTs with and without the high-level interface in [analysis_1](analysis_1.ipynb) and [analysis_2](analysis_2.ipynb) notebooks, respectively
# - Learn about 2D model fitting in the [modeling 2D](modeling_2D.ipynb) notebook
# - find more about Fermi-LAT data analysis in the [fermi_lat](fermi_lat.ipynb) notebook
# - Use source candidates to build a model and perform a 3D fitting (see [analysis_3d](analysis_3d.ipynb), [analysis_mwl](analysis_mwl.ipynb) notebooks for some hints)

# In[ ]:




