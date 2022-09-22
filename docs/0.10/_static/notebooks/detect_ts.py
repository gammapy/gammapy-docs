
# coding: utf-8

# # Source detection with Gammapy
# 
# ## Introduction
# 
# This notebook show how to do source detection with Gammapy using one of the methods available in [gammapy.detect](https://docs.gammapy.org/0.10/detect/index.html).
# 
# We will do this:
# 
# * produce 2-dimensional test-statistics (TS) images using Fermi-LAT 3FHL high-energy Galactic center dataset
# * run a peak finder to make a source catalog
# * do some simple measurements on each source
# * compare to the 3FHL catalog
# 
# Note that what we do here is a quick-look analysis, the production of real source catalogs use more elaborate procedures.
# 
# We will work with the following functions and classes:
# 
# * [gammapy.maps.WcsNDMap](https://docs.gammapy.org/0.10/api/gammapy.maps.WcsNDMap.html)
# * [gammapy.detect.TSMapEstimator](https://docs.gammapy.org/0.10/api/gammapy.detect.TSMapEstimator.html)
# * [gammapy.detect.find_peaks](https://docs.gammapy.org/0.10/api/gammapy.detect.find_peaks.html)

# ## Setup
# 
# As always, let's get started with some setup ...

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


import numpy as np
from astropy import units as u
from gammapy.maps import Map
from gammapy.detect import TSMapEstimator, find_peaks
from gammapy.catalog import source_catalogs
from gammapy.cube import PSFKernel


# ## Read in input images
# 
# We first read in the counts cube and sum over the energy axis:

# In[ ]:


counts = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts.fits.gz")
background = Map.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-background.fits.gz"
)
exposure = Map.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-exposure.fits.gz"
)

maps = {"counts": counts, "background": background, "exposure": exposure}

kernel = PSFKernel.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-psf.fits.gz"
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'estimator = TSMapEstimator()\nimages = estimator.run(maps, kernel.data)')


# ## Plot images

# In[ ]:


plt.figure(figsize=(15, 5))
images["sqrt_ts"].plot(add_cbar=True);


# In[ ]:


plt.figure(figsize=(15, 5))
images["flux"].plot(add_cbar=True, stretch="sqrt", vmin=0);


# In[ ]:


plt.figure(figsize=(15, 5))
images["niter"].plot(add_cbar=True);


# ## Source catalog
# 
# Let's run a peak finder on the `sqrt_ts` image to get a list of sources (positions and peak `sqrt_ts` values).

# In[ ]:


sources = find_peaks(images["sqrt_ts"], threshold=8)
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


# ## Measurements
# 
# * TODO: show cutout for a few sources and some aperture photometry measurements (e.g. energy distribution, significance, flux)

# In[ ]:


# TODO


# ## Compare to 3FHL
# 
# TODO

# In[ ]:


fermi_3fhl = source_catalogs["3fhl"]


# In[ ]:


selection = counts.geom.contains(fermi_3fhl.positions)
fermi_3fhl.table = fermi_3fhl.table[selection]


# In[ ]:


fermi_3fhl.table[["Source_Name", "GLON", "GLAT"]]


# ## Exercises
# 
# TODO: put one or more exercises

# In[ ]:


# Start exercises here!


# ## What next?
# 
# In this notebook, we have seen how to work with images and compute TS images from counts data, if a background estimate is already available.
# 
# Here's some suggestions what to do next:
# 
# - TODO: point to background estimation examples
# - TODO: point to other docs ...
