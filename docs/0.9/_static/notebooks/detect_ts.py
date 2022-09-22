
# coding: utf-8

# # Source detection with Gammapy
# 
# ## Introduction
# 
# This notebook show how to do source detection with Gammapy using one of the methods available in [gammapy.detect](https://docs.gammapy.org/0.9/detect/index.html).
# 
# We will do this:
# 
# * produce 2-dimensional test-statistics (TS) images using Fermi-LAT 2FHL high-energy Galactic plane survey dataset
# * run a peak finder to make a source catalog
# * do some simple measurements on each source
# * compare to the 2FHL catalog
# 
# Note that what we do here is a quick-look analysis, the production of real source catalogs use more elaborate procedures.
# 
# We will work with the following functions and classes:
# 
# * [gammapy.maps.WcsNDMap](https://docs.gammapy.org/0.9/api/gammapy.maps.WcsNDMap.html)
# * [gammapy.detect.TSMapEstimator](https://docs.gammapy.org/0.9/api/gammapy.detect.TSMapEstimator.html)
# * [gammapy.detect.find_peaks](https://docs.gammapy.org/0.9/api/gammapy.detect.find_peaks.html)

# ## Setup
# 
# As always, let's get started with some setup ...

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


import numpy as np
from astropy import units as u
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord
from gammapy.maps import Map
from gammapy.detect import TSMapEstimator, find_peaks
from gammapy.catalog import source_catalogs


# ## Compute TS image

# In[ ]:


# Load data from files
filename = "$GAMMAPY_DATA/fermi_survey/all.fits.gz"
opts = {
    "position": SkyCoord(0, 0, unit="deg", frame="galactic"),
    "width": (20, 8),
}
maps = {
    "counts": Map.read(filename, hdu="COUNTS").cutout(**opts),
    "background": Map.read(filename, hdu="BACKGROUND").cutout(**opts),
    "exposure": Map.read(filename, hdu="EXPOSURE").cutout(**opts),
}


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Compute a source kernel (source template) in oversample mode,\n# PSF is not taken into account\nkernel = Gaussian2DKernel(2.5, mode="oversample")\nestimator = TSMapEstimator()\nimages = estimator.run(maps, kernel)')


# ## Plot images

# In[ ]:


plt.figure(figsize=(15, 5))
images["sqrt_ts"].plot();


# In[ ]:


plt.figure(figsize=(15, 5))
images["flux"].plot(add_cbar=True);


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

images["sqrt_ts"].plot()

plt.gca().scatter(
    sources["ra"],
    sources["dec"],
    transform=plt.gca().get_transform("icrs"),
    color="none",
    edgecolor="black",
    marker="o",
    s=600,
    lw=1.5,
);


# ## Measurements
# 
# * TODO: show cutout for a few sources and some aperture photometry measurements (e.g. energy distribution, significance, flux)

# In[ ]:


# TODO


# ## Compare to 2FHL
# 
# TODO

# In[ ]:


fermi_2fhl = source_catalogs["2fhl"]
fermi_2fhl.table[:5][["Source_Name", "GLON", "GLAT"]]


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
