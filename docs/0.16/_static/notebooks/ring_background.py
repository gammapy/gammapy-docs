#!/usr/bin/env python
# coding: utf-8

# # Ring Background Estimation
# 
# ## Context:
# One of the challenges of IACT analysis is accounting for the large residual hadronic emission. An excess map, assumed to be a map of only gamma-ray events, requires a good estimate of the background.  However, in the absence of a solid template bkg model it is not possible to obtain reliable background model a priori. It was often found necessary in classical cherenkov astronomy to perform a local renormalization of the existing templates, usually with a ring kernel. This assumes that most of the events are background and requires to have an exclusion mask to remove regions with bright signal from the estimation. To read more about this method, see [here.](https://arxiv.org/abs/astro-ph/0610959)
# 
# ## Objective:
# Create an excess (gamma-ray events) map of MSH 15-52 as well as a significance map to determine how solid the signal is.
# 
# ## Proposed approach:
# 
# The analysis workflow is roughly
#  - Compute the sky maps keeping each observation separately using the `Analysis` class
#  - Estimate the background using the `RingBackgroundMaker`
#  - Compute the correlated excess and significance maps using `compute_lima_on_off_image`
#  
# The normalised background thus obtained can be used for general modelling and fitting.

# ## Setup
# As usual, we'll start with some general imports...

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
from regions import CircleSkyRegion
from astropy.convolution import Tophat2DKernel
from scipy.stats import norm

import logging

log = logging.getLogger(__name__)


# Now let's import gammapy specific classes and functions

# In[ ]:


from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.cube import RingBackgroundMaker, MapDatasetOnOff
from gammapy.detect import compute_lima_on_off_image
from gammapy.maps import Map


# ## Creating the config file
# Now, we create a config file for out analysis. You may load this from disc if you have a pre-defined config file.
# 
# In this example, we will use a few HESS runs on the pulsar wind nebula, MSH 1552

# In[ ]:


# source_pos = SkyCoord.from_name("MSH 15-52")
source_pos = SkyCoord(228.32, -59.08, unit="deg")


# In[ ]:


config = AnalysisConfig()
# Select observations - 2.5 degrees from the source position
config.observations.datastore = "$GAMMAPY_DATA/hess-dl3-dr1/"
config.observations.obs_cone = {
    "frame": "icrs",
    "lon": source_pos.ra,
    "lat": source_pos.dec,
    "radius": 2.5 * u.deg,
}


# In[ ]:


config.datasets.type = "3d"
config.datasets.geom.wcs.skydir = {
    "lon": source_pos.ra,
    "lat": source_pos.dec,
    "frame": "icrs",
}  # The WCS geometry - centered on MSH 15-52
config.datasets.geom.wcs.fov = {"width": "3 deg", "height": "3 deg"}
config.datasets.geom.wcs.binsize = "0.02 deg"

# The FoV radius to use for cutouts
config.datasets.geom.selection.offset_max = 3.5 * u.deg

# We now fix the energy axis for the counts map - (the reconstructed energy binning)
config.datasets.geom.axes.energy.min = "0.5 TeV"
config.datasets.geom.axes.energy.max = "5 TeV"
config.datasets.geom.axes.energy.nbins = 10

# We need to extract the ring for each observation separately, hence, no stacking at this stage
config.datasets.stack = False


# In[ ]:


print(config)


# ## Getting the reduced dataset
# We now use the config file to do the initial data reduction which will then be used for a ring extraction

# In[ ]:


get_ipython().run_cell_magic('time', '', '# create the config\nanalysis = Analysis(config)\n\n# for this specific case,w e do not need fine bins in true energy\nanalysis.config.datasets.geom.axes.energy_true = (\n    analysis.config.datasets.geom.axes.energy\n)\n\n# `First get the required observations\nanalysis.get_observations()')


# In[ ]:


print(analysis.config)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Data extraction\nanalysis.get_datasets()')


# ## Extracting the ring background
# 
# Since the ring background is extracted from real off events, we need to use the wstat statistics in this case. For this, we will use the `MapDatasetOnOFF` and the `RingBackgroundMaker` classes.

# ### Create exclusion mask
# First, we need to create an exclusion mask on the known sources. In this case, we need to mask only `MSH 15-52` but this depends on the sources present in our field of view.

# In[ ]:


# get the geom that we use
geom = analysis.datasets[0].counts.geom
energy_axis = analysis.datasets[0].counts.geom.get_axis_by_name("energy")
geom_image = geom.to_image().to_cube([energy_axis.squash()])

# Make the exclusion mask
regions = CircleSkyRegion(center=source_pos, radius=0.3 * u.deg)
exclusion_mask = Map.from_geom(geom_image)
exclusion_mask.data = geom_image.region_mask([regions], inside=False)
exclusion_mask.sum_over_axes().plot();


# For the present analysis, we use a ring with an inner radius of 0.5 deg and width of 0.3 deg.

# In[ ]:


ring_maker = RingBackgroundMaker(
    r_in="0.5 deg", width="0.3 deg", exclusion_mask=exclusion_mask
)


# ### Create a stacked dataset
# Now, we extract the background for each dataset and then stack the maps together to create a single stacked map for further analysis

# In[ ]:


#%%time
stacked_on_off = MapDatasetOnOff.create(geom=geom_image)
for dataset in analysis.datasets:
    dataset_image = (
        dataset.to_image()
    )  # Ring extracting makes sense only for 2D analysis
    dataset_on_off = ring_maker.run(dataset_image)
    stacked_on_off.stack(dataset_on_off)


# This `stacked_on_off` has `on` and `off` counts and acceptance maps which we will use in all further analysis. The `acceptance` and `acceptance_off` maps are the system acceptance of gamma-ray like events in the `on` and `off` regions respectively.

# In[ ]:


print(stacked_on_off)


# ## Compute correlated significance and correlated excess maps
# We need to convolve our maps with an apprpriate smoothing kernel. The significance is computed according to the Li & Ma expression for ON and OFF Poisson measurements, see [here](https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L/abstract). Since astropy convolution kernels only accept integers, we first convert our required size in degrees to int depending on our pixel size.

# In[ ]:


scale = geom.pixel_scales[0].to("deg")
# Using a convolution radius of 0.04 degrees
theta = 0.04 * u.deg / scale
tophat = Tophat2DKernel(theta)
tophat.normalize("peak")

lima_maps = compute_lima_on_off_image(
    stacked_on_off.counts,
    stacked_on_off.counts_off,
    stacked_on_off.acceptance,
    stacked_on_off.acceptance_off,
    tophat,
)


# In[ ]:


significance_map = lima_maps["significance"]
excess_map = lima_maps["excess"]


# In[ ]:


# We can plot the excess and significance maps
plt.figure(figsize=(10, 10))
ax1 = plt.subplot(221, projection=significance_map.geom.wcs)
ax2 = plt.subplot(222, projection=excess_map.geom.wcs)

ax1.set_title("Significance map")
significance_map.get_image_by_idx((0,)).plot(ax=ax1, add_cbar=True)

ax2.set_title("Excess map")
excess_map.get_image_by_idx((0,)).plot(ax=ax2, add_cbar=True)


# It is often important to look at the signficance distribution outside the exclusion region to check that the background estimation is not contaminated by gamma-ray events. This can be the case when exclusion regions are not large enough.
# Typically, we expect the off distribution to be a standard normal distribution.

# In[ ]:


# create a 2D mask for the images
significance_map_off = significance_map * exclusion_mask
significance_all = significance_map.data[np.isfinite(significance_map.data)]
significance_off = significance_map_off.data[
    np.isfinite(significance_map_off.data)
]

plt.hist(
    significance_all,
    density=True,
    alpha=0.5,
    color="red",
    label="all bins",
    bins=21,
)

plt.hist(
    significance_off,
    density=True,
    alpha=0.5,
    color="blue",
    label="off bins",
    bins=21,
)

# Now, fit the off distribution with a Gaussian
mu, std = norm.fit(significance_off)
x = np.linspace(-8, 8, 50)
p = norm.pdf(x, mu, std)
plt.plot(x, p, lw=2, color="black")
plt.legend()
plt.xlabel("Significance")
plt.yscale("log")
plt.ylim(1e-5, 1)
xmin, xmax = np.min(significance_all), np.max(significance_all)
plt.xlim(xmin, xmax)

print(f"Fit results: mu = {mu:.2f}, std = {std:.2f}")


# In[ ]:




