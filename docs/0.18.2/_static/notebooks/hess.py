#!/usr/bin/env python
# coding: utf-8

# # H.E.S.S. data with Gammapy
# 
# [H.E.S.S.](https://www.mpi-hd.mpg.de/hfm/HESS/) is an array of gamma-ray telescopes located in Namibia. Gammapy is regularly used and fully supports H.E.S.S. high-level data analysis, after export to the current [open data level 3 format](https://gamma-astro-data-formats.readthedocs.io/).
# 
# The H.E.S.S. data is private, and H.E.S.S. analysis is mostly documented and discussed at https://hess-confluence.desy.de/ and in H.E.S.S.-internal communication channels. However, in 2018, a small sub-set of archival H.E.S.S. data was publicly released, called the [H.E.S.S. DL3 DR1](https://www.mpi-hd.mpg.de/hfm/HESS/pages/dl3-dr1/), the data level 3, data release number 1. This dataset is 50 MB in size and is used in many Gammapy analysis tutorials, and can be downloaded via [`gammapy download`](https://docs.gammapy.org/dev/scripts/index.html?highlight=download).
# 
# This notebook is a quick introduction to H.E.S.S. data and instrument responses and contains some specifics that are important for H.E.S.S. users:
# 
# - IRF formats and shapes
# - How to handle safe energy and max offset
# - EVENTS and GTI formats (e.g. how HESS 1, 2, configs, ... are handled)
# - Link to HESS Confluence where data and help is available (Slack channel)?
# 
# Then at the end, link to other analysis tutorials that are likely of interest for H.E.S.S. people, and add a few exercises. This can be short, a 5-10 min read. It's just supposed to be the "landing page" for someone new in H.E.S.S. that has never used Gammapy.
# 
# ## DL3 DR1
# 
# This is how to access data and IRFs from the H.E.S.S. data level 3, data release 1.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u


# In[ ]:


from gammapy.data import DataStore
from gammapy.maps import MapAxis
from gammapy.makers.utils import make_theta_squared_table
from gammapy.visualization import plot_theta_squared_table


# In[ ]:


data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")


# In[ ]:


data_store.info()


# In[ ]:


data_store.obs_table[:2][["OBS_ID", "DATE-OBS", "RA_PNT", "DEC_PNT", "OBJECT"]]


# In[ ]:


obs = data_store.obs(23523)


# In[ ]:


obs.events.select_offset([0, 2.5] * u.deg).peek()


# In[ ]:


obs.aeff.peek()


# In[ ]:


obs.edisp.peek()


# In[ ]:


obs.psf.peek()


# In[ ]:


obs.bkg.to_2d().plot()


# ## Theta squared event distribution
# As a quick look plot it can be helpful to plot the quadratic offset (theta squared) distribution of the events. 

# In[ ]:


position = SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs")
theta2_axis = MapAxis.from_bounds(0, 0.2, nbin=20, interp="lin", unit="deg2")

observations = data_store.get_observations([23523, 23526])
theta2_table = make_theta_squared_table(
    observations=observations,
    position=position,
    theta_squared_axis=theta2_axis,
)


# In[ ]:


plt.figure(figsize=(10, 5))
plot_theta_squared_table(theta2_table)


# ## Exercises
# 
# - Find the `OBS_ID` for the runs of the Crab nebula
# - Compute the expected number of background events in the whole FOV for `OBS_ID=23523` in the 1 TeV to 3 TeV energy band, from the background IRF.

# ## Next steps
# 
# Now you know how to access and work with H.E.S.S. data. All other tutorials and documentation apply to H.E.S.S. and CTA or any other IACT that provides DL3 data and IRFs in the standard format.
