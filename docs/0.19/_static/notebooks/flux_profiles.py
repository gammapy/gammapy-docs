#!/usr/bin/env python
# coding: utf-8

# # Flux Profile Estimation
# 
# 
# This tutorial shows how to estimate flux profiles.
# 
# ## Prerequisites
# 
# Knowledge of 3D data reduction and datasets used in Gammapy, see for instance the first analysis tutorial.
# 
# 
# ## Context
# A useful tool to study and compare the saptial distribution of flux in images and data cubes is the measurement of flxu profiles. Flux profiles can show spatial correlations of gamma-ray data with e.g. gas maps or other type of gamma-ray data. Most commonly flux profiles are measured along some preferred coordinate axis, either radially distance from a source of interest, along longitude and latitude coordinate axes or along the path defined by two spatial coordinates. 
# 
# ## Proposed Approach
# Flux profile estimation essentially works by estimating flux points for a set of predefined spatially connected regions. For radial flux profiles the shape of the regions are annuli with a common center, for linear profiles it's typically a rectangular shape.
# 
# We will work on a pre-computed `MapDataset` of Fermi-LAT data, use `Region` to define the structure of the bins of the flux profile and run the actually profile extraction using the `FluxProfileEstimator` 
# 

# ## Setup and Imports

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np


# In[ ]:


from gammapy.datasets import MapDataset
from gammapy.estimators import FluxProfileEstimator, FluxPoints
from gammapy.maps import RegionGeom
from gammapy.utils.regions import (
    make_concentric_annulus_sky_regions,
    make_orthogonal_rectangle_sky_regions,
)
from gammapy.modeling.models import PowerLawSpectralModel


# ## Read and Introduce Data

# In[ ]:


dataset = MapDataset.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc.fits.gz", name="fermi-dataset"
)


# This is what the counts image we will work with looks like:

# In[ ]:


counts_image = dataset.counts.sum_over_axes()
counts_image.smooth("0.1 deg").plot(stretch="sqrt");


# There are 400x200 pixels in the dataset and 11 energy bins between 10 GeV and 2 TeV:

# In[ ]:


print(dataset.counts)


# ## Profile Estimation
# 
# ### Configuration
# 
# We start by defining a list of spatially connected regions along the galactic longitude axis. For this there is a helper function `make_orthogonal_rectangle_sky_regions`. The individual region bins for the profile have a height of 3 deg and in total there are 31 bins. The starts from  lon = 10 deg tand goes to lon = 350 deg. In addition we have to specify the `wcs` to take into account possible projections effects on the region definition:

# In[ ]:


regions = make_orthogonal_rectangle_sky_regions(
    start_pos=SkyCoord("10d", "0d", frame="galactic"),
    end_pos=SkyCoord("350d", "0d", frame="galactic"),
    wcs=counts_image.geom.wcs,
    height="3 deg",
    nbin=51,
)


# We can use the `RegionGeom` object to illustrate the regions on top of the counts image:

# In[ ]:


geom = RegionGeom.create(region=regions)
ax = counts_image.smooth("0.1 deg").plot(stretch="sqrt")
geom.plot_region(ax=ax, color="w");


# Next we create the `FluxProfileEstimator`. For the estimation of the flux profile we assume a spectral model with a power-law shape and an index of 2.3

# In[ ]:


flux_profile_estimator = FluxProfileEstimator(
    regions=regions,
    spectrum=PowerLawSpectralModel(index=2.3),
    energy_edges=[10, 2000] * u.GeV,
    selection_optional=["ul"],
)


# We can see the full configuration by printing the estimator object:

# In[ ]:


print(flux_profile_estimator)


# ### Run Estimation
# Now we can run the profile estimation and explore the results:

# In[ ]:


get_ipython().run_cell_magic('time', '', 'profile = flux_profile_estimator.run(datasets=dataset)')


# In[ ]:


print(profile)


# We can see the flux profile is represented by a `FluxPoints` object with a `projected-distance` axis, which defines the main axis the flux profile is measured along. The `lon` and `lat` axes can be ignored. 
# 
# ### Plotting Results
# 
# Let us directly plot the result using `FluxPoints.plot()`:

# In[ ]:


ax = profile.plot(sed_type="dnde")
ax.set_yscale("linear")


# Based on the spectral model we specified above we can also plot in any other sed type, e.g. energy flux and define a different threshold when to plot upper limits:

# In[ ]:


profile.sqrt_ts_threshold_ul = 2

ax = profile.plot(sed_type="eflux")
ax.set_yscale("linear")


# We can also plot any other quantity of interest, that is defined on the `FluxPoints` result object. E.g. the predicted total counts, background counts and excess counts: 

# In[ ]:


quantities = ["npred", "npred_excess", "npred_background"]

for quantity in quantities:
    profile[quantity].plot(label=quantity.title())

plt.ylabel("Counts ")


# ### Serialisation and I/O
# 
# The profile can be serialised using `FluxPoints.write()`, given a specific format:

# In[ ]:


profile.write(
    filename="flux_profile_fermi.fits",
    format="profile",
    overwrite=True,
    sed_type="dnde",
)


# In[ ]:


profile_new = FluxPoints.read(
    filename="flux_profile_fermi.fits", format="profile"
)


# In[ ]:


ax = profile_new.plot()
ax.set_yscale("linear")


# The profile can be serialised to a `~astropy.table.Table` object using:

# In[ ]:


table = profile.to_table(format="profile", formatted=True)
table


# No we can also estimate a radial profile starting from the Galactic center:

# In[ ]:


regions = make_concentric_annulus_sky_regions(
    center=SkyCoord("0d", "0d", frame="galactic"),
    radius_max="1.5 deg",
    nbin=11,
)


# Again we first illustrate the regions:

# In[ ]:


geom = RegionGeom.create(region=regions)
gc_image = counts_image.cutout(
    position=SkyCoord("0d", "0d", frame="galactic"), width=3 * u.deg
)
ax = gc_image.smooth("0.1 deg").plot(stretch="sqrt")
geom.plot_region(ax=ax, color="w");


# This time we define two energy bins and include the fit statistic profile in the computation:

# In[ ]:


flux_profile_estimator = FluxProfileEstimator(
    regions=regions,
    spectrum=PowerLawSpectralModel(index=2.3),
    energy_edges=[10, 100, 2000] * u.GeV,
    selection_optional=["ul", "scan"],
    norm_values=np.linspace(-1, 5, 11),
)


# In[ ]:


profile = flux_profile_estimator.run(datasets=dataset)


# We can directly plot the result:

# In[ ]:


ax = profile.plot(axis_name="projected-distance", sed_type="flux")


# However because of the powerlaw spectrum the flux at high energies is much lower. To extract the profile at high energies only we can use:

# In[ ]:


profile_high = profile.slice_by_idx({"energy": slice(1, 2)})


# And now plot the points together with the likelihood profiles:

# In[ ]:


ax = profile_high.plot(sed_type="eflux", color="tab:orange")
profile_high.plot_ts_profiles(ax=ax, sed_type="eflux")
ax.set_yscale("linear")


# In[ ]:




