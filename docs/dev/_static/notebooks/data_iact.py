
# coding: utf-8

# # IACT DL3 data with Gammapy
# 
# ## Introduction
# 
# This tutorial will show you how to work with IACT (Imaging Atmospheric Cherenkov Telescope) DL3 ("data level 3").
# 
# We will work with event data and instrument response functions (IRFs), mainly using [gammapy.data](http://docs.gammapy.org/en/latest/data/index.html) and [gammapy.irf](http://docs.gammapy.org/en/latest/irf/index.html).
# 
# This notebook uses a preliminary small test dataset from the CTA first data challenge (1DC).
# 
# The main class to load data is
# 
# * [gammapy.data.DataStore](http://docs.gammapy.org/en/latest/api/gammapy.data.DataStore.html)
# 
# The `DataStore` has two index tables:
# 
# * `DataStore.obs_table` ([gammapy.data.ObservationTable](http://docs.gammapy.org/en/latest/api/gammapy.data.ObservationTable.html)) to list and select available observations.
# * `DataStore.hdu_table` ([gammapy.data.HDUIndexTable](http://docs.gammapy.org/en/latest/api/gammapy.data.HDUIndexTable.html)) to locate data for a given observation.
# 
# Data loading is done via the `DataStore.obs` method which returns a
# 
# * [gammapy.data.DataStoreObservation](http://docs.gammapy.org/en/latest/api/gammapy.data.DataStoreObservation.html)
# 
# object, which on property access loads the data and IRFs and returns them as Gammapy objects.
# 
# We support the common IACT DL3 data formats: http://gamma-astro-data-formats.readthedocs.io/ .
# 
# In this tutorial we will use objects of these types:
# 
# * [gammapy.data.EventList](http://docs.gammapy.org/en/latest/api/gammapy.data.EventList.html)
# 
# 
# * Load [gammapy.irf.EffectiveAreaTable2D](http://docs.gammapy.org/en/latest/api/gammapy.irf.EffectiveAreaTable2D.html), which has AEFF info for the whole field of view (FOV).
# * For the given source offset in the FOV, slice out [gammapy.irf.EffectiveAreaTable](http://docs.gammapy.org/en/latest/api/gammapy.irf.EffectiveAreaTable.html)
# 
# 
# * Load [gammapy.irf.EnergyDispersion2D](http://docs.gammapy.org/en/latest/api/gammapy.irf.EnergyDispersion2D.html), which has EDISP info for the whole FOV.
# * For a given source offset in the FOV, slice out [gammapy.irf.EnergyDispersion](http://docs.gammapy.org/en/latest/api/gammapy.irf.EnergyDispersion.html)
# 
# 
# * Load [gammapy.irf.EnergyDependentMultiGaussPSF](http://docs.gammapy.org/en/latest/api/gammapy.irf.EnergyDependentMultiGaussPSF.html), which has PSF info for the whole FOV using an analytical PSF model.
# * For a given source offset in the FOV, slice out [gammapy.irf.EnergyDependentTablePSF](http://docs.gammapy.org/en/latest/api/gammapy.irf.EnergyDependentTablePSF.html).
# * For a given energy or energy band, compute [gammapy.irf.TablePSF](http://docs.gammapy.org/en/latest/api/gammapy.irf.TablePSF.html).
# 
# 
# 
# 
# ## Setup

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


import numpy as np
import matplotlib
print('numpy : {}'.format(np.__version__))
print('matplotlib : {}'.format(matplotlib.__version__))


# In[3]:


# We only need to import the `DataStore`,
# all other data objects can be loaded via the data store.
from gammapy.data import DataStore
import astropy.units as u


# ## Data store
# 
# First, we need to select some observations for our spectral analysis. To this end we use the [data management](http://docs.gammapy.org/en/latest/data/dm.html) functionality in gammapy. The following example uses a simulated crab dataset in [gammapy-extra](https://github.com/gammapy/gammapy-extra). Ideally, we'd use crabs runs from the H.E.S.S. public data release, so if you have the released files just change the ``DATA_DIR`` variable to the corresponding folder.

# In[4]:


# data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
data_store = DataStore.from_dir('$GAMMAPY_EXTRA/test_datasets/cta_1dc')


# In[5]:


data_store.info()


# In[6]:


print(data_store.hdu_table.colnames)
data_store.hdu_table[:7]


# ## Observation selection
# 
# Select observations using the observation table

# In[7]:


table = data_store.obs_table
print(table.colnames)

subtable = table[
    (np.abs(table['GLAT_PNT']) < 0.2) &
    (table['LIVETIME'] > 1500)
]
print("Found {} runs".format(len(subtable)))
subtable[::100][['OBS_ID', 'GLON_PNT', 'GLAT_PNT', 'LIVETIME']]


# In[8]:


# In the following examples we'll just use this one observation
obs = data_store.obs(obs_id=659)
print(obs)


# ## Events
# 
# Explore the `EventList`

# In[9]:


events = obs.events
events.peek()
events.table[:5][['EVENT_ID', 'TIME', 'ENERGY', 'RA', 'DEC']]


# ## Effective area
# 
# Explore `EffectiveAreaTable2d` and `EffectiveAreaTable`

# In[10]:


aeff = obs.aeff
aeff.peek()
print(aeff)


# In[11]:


# Slice out effective area at a given offset
effarea = aeff.to_effective_area_table(offset=0.5 * u.deg)
effarea.plot()


# ## Energy dispersion
# 
# Explore `EnergyDispersion2d` and `EnergyDispersion`

# In[12]:


edisp = obs.edisp
edisp.peek()
print(edisp)


# In[13]:


# Calculate energy dispersion matrix at a given offset
response_matrix = edisp.to_energy_dispersion(offset=0.5 * u.deg)
response_matrix.plot_bias()


# ## PSF
# 
# TODO: examples for point spread function (PSF)

# In[14]:


psf = obs.psf
psf.peek()
print(psf)


# ## Background model
# 
# TODO: example how to load and plot `gammapy.background.FOVBackgroundModel`

# In[17]:


# TODO: doesn't work yet
# bkg = obs.bkg
# bkg.peek()
# print(bkg)


# ## Exercises
# 
# - TODO

# In[16]:


# Start the exercises here!


# ## What next?
# 
# In this tutorial we have learned how to access and check IACT DL3 data.
# 
# Usually for a science analysis, if others have checked the data and IRF quality for you and you trust it's good, you don't need to do that.
# Instead, you'll just run an analysis and look at higher-level results, like images or spectra.
# 
# Next you could do:
# 
# * image analysis
# * spectral analysis
# * cube analysis
# * time analysis
# * source detection
