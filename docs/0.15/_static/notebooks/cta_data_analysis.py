#!/usr/bin/env python
# coding: utf-8

# # CTA data analysis with Gammapy
# 
# ## Introduction
# 
# **This notebook shows an example how to make a sky image and spectrum for simulated CTA data with Gammapy.**
# 
# The dataset we will use is three observation runs on the Galactic center. This is a tiny (and thus quick to process and play with and learn) subset of the simulated CTA dataset that was produced for the first data challenge in August 2017.
# 

# ## Setup
# 
# As usual, we'll start with some setup ...

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().system('gammapy info --no-envvar --no-system')


# In[ ]:


import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel
from regions import CircleSkyRegion
from gammapy.modeling import Fit, Datasets
from gammapy.data import DataStore
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.spectrum import (
    SpectrumDatasetMaker,
    SpectrumDataset,
    FluxPointsEstimator,
    FluxPointsDataset,
    ReflectedRegionsBackgroundMaker,
    plot_spectrum_datasets_off_regions,
)
from gammapy.maps import MapAxis, WcsNDMap, WcsGeom
from gammapy.cube import MapDatasetMaker, MapDataset, SafeMaskMaker
from gammapy.detect import TSMapEstimator, find_peaks


# In[ ]:


# Configure the logger, so that the spectral analysis
# isn't so chatty about what it's doing.
import logging

logging.basicConfig()
log = logging.getLogger("gammapy.spectrum")
log.setLevel(logging.ERROR)


# ## Select observations
# 
# A Gammapy analysis usually starts by creating a `~gammapy.data.DataStore` and selecting observations.
# 
# This is shown in detail in the other notebook, here we just pick three observations near the galactic center.

# In[ ]:


data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps")


# In[ ]:


# Just as a reminder: this is how to select observations
# from astropy.coordinates import SkyCoord
# table = data_store.obs_table
# pos_obs = SkyCoord(table['GLON_PNT'], table['GLAT_PNT'], frame='galactic', unit='deg')
# pos_target = SkyCoord(0, 0, frame='galactic', unit='deg')
# offset = pos_target.separation(pos_obs).deg
# mask = (1 < offset) & (offset < 2)
# table = table[mask]
# table.show_in_browser(jsviewer=True)


# In[ ]:


obs_id = [110380, 111140, 111159]
observations = data_store.get_observations(obs_id)


# In[ ]:


obs_cols = ["OBS_ID", "GLON_PNT", "GLAT_PNT", "LIVETIME"]
data_store.obs_table.select_obs_id(obs_id)[obs_cols]


# ## Make sky images
# 
# ### Define map geometry
# 
# Select the target position and define an ON region for the spectral analysis

# In[ ]:


axis = MapAxis.from_edges(
    np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(0, 0), npix=(500, 400), binsz=0.02, coordsys="GAL", axes=[axis]
)
geom


# ### Compute images
# 
# Exclusion mask currently unused. Remove here or move to later in the tutorial?

# In[ ]:


target_position = SkyCoord(0, 0, unit="deg", frame="galactic")
on_radius = 0.2 * u.deg
on_region = CircleSkyRegion(center=target_position, radius=on_radius)


# In[ ]:


exclusion_mask = geom.to_image().region_mask([on_region], inside=False)
exclusion_mask = WcsNDMap(geom.to_image(), exclusion_mask)
exclusion_mask.plot();


# In[ ]:


get_ipython().run_cell_magic('time', '', 'stacked = MapDataset.create(geom=geom)\nmaker = MapDatasetMaker(selection=["counts", "background", "exposure"])\nmaker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=2.5 * u.deg)\n\nfor obs in observations:\n    cutout = stacked.cutout(obs.pointing_radec, width="5 deg")\n    dataset = maker.run(cutout, obs)\n    dataset = maker_safe_mask.run(dataset, obs)\n    stacked.stack(dataset)')


# In[ ]:


# The maps are cubes, with an energy axis.
# Let's also make some images:
dataset_image = stacked.to_image()

images = {
    "counts": dataset_image.counts.get_image_by_idx((0,)),
    "exposure": dataset_image.exposure.get_image_by_idx((0,)),
    "background": dataset_image.background_model.map.get_image_by_idx((0,)),
}

images["excess"] = images["counts"] - images["background"]


# ### Show images
# 
# Let's have a quick look at the images we computed ...

# In[ ]:


images["counts"].smooth(2).plot(vmax=5);


# In[ ]:


images["background"].plot(vmax=5);


# In[ ]:


images["excess"].smooth(3).plot(vmax=2);


# ## Source Detection
# 
# Use the class `~gammapy.detect.TSMapEstimator` and `~gammapy.detect.find_peaks` to detect sources on the images:

# In[ ]:


kernel = Gaussian2DKernel(1, mode="oversample").array
plt.imshow(kernel);


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ts_image_estimator = TSMapEstimator()\nimages_ts = ts_image_estimator.run(images, kernel)\nprint(images_ts.keys())')


# In[ ]:


sources = find_peaks(images_ts["sqrt_ts"], threshold=8)
sources


# In[ ]:


source_pos = SkyCoord(sources["ra"], sources["dec"])
source_pos


# In[ ]:


# Plot sources on top of significance sky image
images_ts["sqrt_ts"].plot(add_cbar=True)

plt.gca().scatter(
    source_pos.ra.deg,
    source_pos.dec.deg,
    transform=plt.gca().get_transform("icrs"),
    color="none",
    edgecolor="white",
    marker="o",
    s=200,
    lw=1.5,
);


# ## Spatial analysis
# 
# See other notebooks for how to run a 3D cube or 2D image based analysis.

# ## Spectrum
# 
# We'll run a spectral analysis using the classical reflected regions background estimation method,
# and using the on-off (often called WSTAT) likelihood function.

# In[ ]:


e_reco = np.logspace(-1, np.log10(40), 40) * u.TeV
e_true = np.logspace(np.log10(0.05), 2, 200) * u.TeV

dataset_empty = SpectrumDataset.create(
    e_reco=e_reco, e_true=e_true, region=on_region
)


# In[ ]:


dataset_maker = SpectrumDatasetMaker(
    containment_correction=False, selection=["counts", "aeff", "edisp"]
)
bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'datasets = []\n\nfor observation in observations:\n    dataset = dataset_maker.run(dataset_empty, observation)\n    dataset_on_off = bkg_maker.run(dataset, observation)\n    dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)\n    datasets.append(dataset_on_off)')


# In[ ]:


plt.figure(figsize=(8, 8))
_, ax, _ = images["counts"].smooth("0.03 deg").plot(vmax=8)

on_region.to_pixel(ax.wcs).plot(ax=ax, edgecolor="white")
plot_spectrum_datasets_off_regions(datasets, ax=ax)


# ### Model fit
# 
# The next step is to fit a spectral model, using all data (i.e. a "global" fit, using all energies).

# In[ ]:


get_ipython().run_cell_magic('time', '', 'spectral_model = PowerLawSpectralModel(\n    index=2, amplitude=1e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV\n)\nmodel = SkyModel(spectral_model=spectral_model)\nfor dataset in datasets:\n    dataset.models = model\n\nfit = Fit(datasets)\nresult = fit.run()\nprint(result)')


# ### Spectral points
# 
# Finally, let's compute spectral points. The method used is to first choose an energy binning, and then to do a 1-dim likelihood fit / profile to compute the flux and flux error.

# In[ ]:


# Flux points are computed on stacked observation
stacked_dataset = Datasets(datasets).stack_reduce()

print(stacked_dataset)


# In[ ]:


e_edges = np.logspace(0, 1.5, 5) * u.TeV

stacked_dataset.model = model

fpe = FluxPointsEstimator(datasets=[stacked_dataset], e_edges=e_edges)
flux_points = fpe.run()
flux_points.table_formatted


# ### Plot
# 
# Let's plot the spectral model and points. You could do it directly, but there is a helper class.
# Note that a spectral uncertainty band, a "butterfly" is drawn, but it is very thin, i.e. barely visible.

# In[ ]:


model.spectral_model.parameters.covariance = result.parameters.covariance
flux_points_dataset = FluxPointsDataset(data=flux_points, models=model)


# In[ ]:


plt.figure(figsize=(8, 6))
flux_points_dataset.peek();


# ## Exercises
# 
# * Re-run the analysis above, varying some analysis parameters, e.g.
#     * Select a few other observations
#     * Change the energy band for the map
#     * Change the spectral model for the fit
#     * Change the energy binning for the spectral points
# * Change the target. Make a sky image and spectrum for your favourite source.
#     * If you don't know any, the Crab nebula is the "hello world!" analysis of gamma-ray astronomy.

# In[ ]:


# print('hello world')
# SkyCoord.from_name('crab')


# ## What next?
# 
# * This notebook showed an example of a first CTA analysis with Gammapy, using simulated 1DC data.
# * Let us know if you have any question or issues!
