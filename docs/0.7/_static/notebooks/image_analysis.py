
# coding: utf-8

# # Image analysis with Gammapy
# 
# ## Introduction
# 
# This tutorial shows how to make a significance image of the Crab nebula with Gammapy.
# 
# * Use the [gammapy.data.DataStore](http://docs.gammapy.org/0.7/api/gammapy.data.DataStore.html)
#   to load [gammapy.data.EventList](http://docs.gammapy.org/0.7/api/gammapy.data.EventList.html) data.
# * Fill the events in a [gammapy.image.SkyImage](http://docs.gammapy.org/0.7/api/gammapy.image.SkyImage.html)
# 
# 
# * We'll use the [astropy.convolution.Tophat2DKernel](http://docs.astropy.org/en/latest/api/astropy.convolution.Tophat2DKernel.html) and 
#  [astropy.convolution.Ring2DKernel](http://docs.astropy.org/en/latest/api/astropy.convolution.Ring2DKernel.html) kernels and the [gammapy.detect.KernelBackgroundEstimator](http://docs.gammapy.org/0.7/api/gammapy.detect.KernelBackgroundEstimator.html) to estimate the background.
# * Run [gammapy.scripts.StackedObsImageMaker](http://docs.gammapy.org/0.7/api/gammapy.scripts.StackedObsImageMaker.html) to get images and PSF.
# 
# 
# TODO: Refactor [gammapy.scripts.image_fit](http://docs.gammapy.org/0.7/api/gammapy.scripts.image_fit.html) into a class (simiar to `gammapy.spectrum.SpectrumFit`) and run it here to fit a Gauss and get the position / extension.

# ## Setup
# 
# As usual, IPython notebooks start with some setup and Python imports

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Ring2DKernel, Tophat2DKernel
from astropy.visualization import simple_norm

from gammapy.data import DataStore
from gammapy.image import SkyImage, SkyImageList
from gammapy.detect import KernelBackgroundEstimator as KBE


# ## Dataset
# 
# We will use the `gammapy.data.DataStore` to access some example data.
# 
# These are observations of the Crab nebula with H.E.S.S. (preliminary, events are simulated for now).

# In[3]:


data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')


# ## Counts image
# 
# Let's make a counts image using the `SkyMap` class.

# In[4]:


source_pos = SkyCoord(83.633083, 22.0145, unit='deg')
# If you have internet access, you could also use this to define the `source_pos`:
# source_pos = SkyCoord.from_name('crab')
print(source_pos)


# In[5]:


ref_image = SkyImage.empty(
    nxpix=400, nypix=400, binsz=0.02,
    xref=source_pos.ra.deg, yref=source_pos.dec.deg,
    coordsys='CEL', proj='TAN',
)


# In[6]:


# Make a counts image for a single observation
events = data_store.obs(obs_id=23523).events
counts_image = SkyImage.empty_like(ref_image)
counts_image.fill_events(events)


# In[7]:


norm = simple_norm(counts_image.data, stretch='sqrt', min_cut=0, max_cut=0.3)
counts_image.smooth(radius=0.1 * u.deg).plot(norm=norm, add_cbar=True)


# In[8]:


# Making a counts image for multiple observations is a bit inconvenient at the moment
# we'll make that better soon.
# For now, you can do it like this:
obs_ids = [23523, 23526]
counts_image2 = SkyImage.empty_like(ref_image)
for obs_id in obs_ids:
    events = data_store.obs(obs_id=obs_id).events
    counts_image2.fill_events(events)


# In[9]:


norm = simple_norm(counts_image2.data, stretch='sqrt', min_cut=0, max_cut=0.5)
counts_image2.smooth(radius=0.1 * u.deg).plot(norm=norm, add_cbar=True)


# # Background modeling
# 
# In Gammapy a few different methods to estimate the background are available.
# 
# Here we'll use the [gammapy.detect.KernelBackgroundEstimator](http://docs.gammapy.org/0.7/api/gammapy.detect.KernelBackgroundEstimator.html) to make a background image
# and the make a significance image.

# In[10]:


source_kernel = Tophat2DKernel(radius=5)
source_kernel.normalize(mode='peak')
source_kernel = source_kernel.array

background_kernel = Ring2DKernel(radius_in=20, width=10)
background_kernel.normalize(mode='peak')
background_kernel = background_kernel.array


# In[11]:


plt.imshow(source_kernel, interpolation='nearest', cmap='gray')
plt.colorbar()
plt.grid('off')


# In[12]:


plt.imshow(background_kernel, interpolation='nearest', cmap='gray')
plt.colorbar()
plt.grid('off')


# In[13]:


# To use the `KernelBackgroundEstimator` you first have to set
# up a source and background kernel and put the counts image input
# into a container `SkyImageList` class.
images = SkyImageList()
images['counts'] = counts_image2

kbe = KBE(
    kernel_src=source_kernel,
    kernel_bkg=background_kernel,
    significance_threshold=5,
    mask_dilation_radius=0.06 * u.deg,
)
# This takes about 10 seconds on my machine
result = kbe.run(images)


# In[14]:


# Let's have a look at the background image and the exclusion mask

# This doesn't work yet ... need to do SkyImage.plot fixes:
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
# background_image.plot(ax=axes[0])
# exclusion_image.plot(ax=axes[1])
# significance_image.plot(ax=axes[2])


# In[15]:


background_image = result['background']
norm = simple_norm(background_image.data, stretch='sqrt', min_cut=0, max_cut=0.5)
background_image.plot(norm=norm, add_cbar=True)


# In[16]:


result['exclusion'].plot()


# In[17]:


significance_image = result['significance']
significance_image.plot(add_cbar=True, vmin=-3, vmax=20)


# ## Morphology fit
# 
# TODO

# In[18]:


# from gammapy.scripts import image_fit


# ## Exercises
# 
# - Compute the counts, excess, background and significance at the Crab nebula position.
# - Make an energy distribution of the events at the Crab nebula position.

# ## What next?
# 
# TODO: summarise
# 
# TODO: give links what to do next.
