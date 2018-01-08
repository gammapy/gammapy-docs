
# coding: utf-8

# # Simulating and analysing sources and diffuse emission
# 
# ## Introduction
# 
# This notebook shows how to do the analysis presented in this paper using Gammapy:
# 
# * Ellis Owen et al. (2015). *"The $\gamma$-ray Milky Way above 10 GeV: Distinguishing Sources from Diffuse Emission"*  ([ADS](http://adsabs.harvard.edu/abs/2015arXiv150602319O))
# 
# The following parts of Gammapy are used:
# 
# * [gammapy.image.SkyImage](http://docs.gammapy.org/en/latest/api/gammapy.image.SkyImage.html)
# * [gammapy.detect.KernelBackgroundEstimator](http://docs.gammapy.org/en/latest/api/gammapy.detect.KernelBackgroundEstimator.html)
# * TODO: code to simulate images from a source catalog is where?!?
# 
# 
# ## Setup

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[ ]:


import logging
logging.basicConfig(level=logging.INFO)
from astropy.io import fits
from scipy.ndimage import convolve
from gammapy.stats import significance
# from gammapy.image import SkyImage
from gammapy.detect import KernelBackgroundEstimator


# ## Simulation
# 
# TODO

# In[ ]:


# Parameters
TOTAL_COUNTS = 1e6
SOURCE_FRACTION = 0.2

CORRELATION_RADIUS = 0.1 # deg
SIGNIFICANCE_THRESHOLD = 4.
MASK_DILATION_RADIUS = 0.2 # deg
NUMBER_OF_ITERATIONS = 3

# Derived parameters
DIFFUSE_FRACTION = 1. - SOURCE_FRACTION


# In[ ]:


# Load example model images
source_image_true = fits.getdata('sources.fits.gz')
diffuse_image_true = fits.getdata('diffuse.fits.gz')


# In[ ]:


# Generate example data
source_image_true *= SOURCE_FRACTION * TOTAL_COUNTS / source_image_true.sum()
diffuse_image_true *= DIFFUSE_FRACTION * TOTAL_COUNTS / diffuse_image_true.sum()
total_image_true = source_image_true + diffuse_image_true

counts = np.random.poisson(total_image_true)

print('source counts: {0}'.format(source_image_true.sum()))
print('diffuse counts: {0}'.format(diffuse_image_true.sum()))


# ## Analysis
# 
# TODO

# In[ ]:


# Start with flat background estimate
background=np.ones_like(counts, dtype=float)
images = KernelBackgroundEstimatorData(counts=counts, background=background)

# CORRELATION_RADIUS
source_kernel = np.ones((5, 5))
background_kernel = np.ones((100, 10))

kbe = KernelBackgroundEstimator(
    images=images,
    source_kernel=source_kernel,
    background_kernel=background_kernel,
    significance_threshold=SIGNIFICANCE_THRESHOLD,
    mask_dilation_radius=MASK_DILATION_RADIUS
)

kbe.run(n_iterations=3)


# In[ ]:


kbe.run_iteration()


# In[ ]:


kbe._data[1].print_info()


# ## Plot results
# 
# TODO

# In[ ]:


n_iterations = 5

#plt.clf()
plt.figure(figsize=(10, 10))

for iteration in range(n_iterations):
    filename = 'test{0:02d}_mask.fits'.format(iteration)
    mask = fits.getdata(filename)[100:300,:]

    plt.subplot(n_iterations, 2, 2 * iteration + 1)
    filename = 'test{0:02d}_background.fits'.format(iteration)
    data = fits.getdata(filename)[100:300,:]
    plt.imshow(data)
    plt.contour(mask, levels=[0], linewidths=2, colors='white')
    plt.axis('off')
    plt.title(filename)
    
    plt.subplot(n_iterations, 2, 2 * iteration + 2)
    filename = 'test{0:02d}_significance.fits'.format(iteration)
    data = fits.getdata(filename)[100:300,:]
    plt.imshow(data, vmin=-3, vmax=5)
    plt.contour(mask, levels=[0], linewidths=2, colors='white')
    plt.axis('off')
    plt.title(filename)
    #plt.colorbar()

#plt.tight_layout()


# ## Exercises
# 
# - Vary some parameter and check how results change
# - Apply same method to 2FHL or some other real dataset?

# ## What next?
# 
# TODO: summarise
# 
# TODO: pointers to other docs
