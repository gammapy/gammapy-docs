
# coding: utf-8

# # Template background model production with Gammapy
# 
# ## Introduction 
# 
# In this tutorial, we will create a template background model in the `bkg_2d` format, i.e. with offset and energy axes (see [spec](http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/background/index.html#bkg-2d-format)).
# 
# We will be working with only 4 H.E.S.S. runs on the Crab nebula here, just as an example.
# 
# To build a coherent background model you normally use 100s of runs of AGN observations or intentional "off" runs that point at parts of the sky containing no known gamma-ray sources.
# 
# We will mainly be using the following classes:
#         
# * [gammapy.data.DataStore](http://docs.gammapy.org/0.7/api/gammapy.data.DataStore.html) to load the runs to use to build the bkg model.
# * [gammapy.data.ObservationGroupAxis](http://docs.gammapy.org/0.7/api/gammapy.data.ObservationGroupAxis.html) and [gammapy.data.ObservationGroups](http://docs.gammapy.org/0.7/api/gammapy.data.ObservationGroups.html) to group the runs
# * [gammapy.background.OffDataBackgroundMaker](http://docs.gammapy.org/0.7/api/gammapy.background.OffDataBackgroundMaker.html) to compute the background model
# * [gammapy.background.EnergyOffsetBackgroundModel](http://docs.gammapy.org/0.7/api/gammapy.background.EnergyOffsetBackgroundModel.html) to represent and write the background model
# 
# 

# ## Setup
# 
# As always, we start the notebook with some setup and imports.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


import shutil
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle


# In[3]:


from gammapy.extern.pathlib import Path
from gammapy.utils.energy import EnergyBounds
from gammapy.utils.nddata import sqrt_space
from gammapy.data import DataStore, ObservationGroupAxis, ObservationGroups
from gammapy.background import EnergyOffsetBackgroundModel
from gammapy.background import OffDataBackgroundMaker
from gammapy.catalog import SourceCatalogGammaCat


# ## Compute background model
# 
# Computing a set of template background model has two major steps:
# 1. Define group of runs for each background model
# 2. Run the `OffDataBackgroundMaker`
# 
# We also need a scratch directory, and a table of known gamma-ray sources to exclude.
# 
# ### Make a scratch directory
# 
# Background model production is a little pipeline that needs a "scratch" directory to put some files while running. Let's make ourselves a fresh empty scratch sub-directory called `background` in the current working directory.

# In[4]:


def make_fresh_dir(path):
    """Make a fresh directory. Delete first if exists"""    
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(str(path))
    path.mkdir()
    return path


# In[5]:


scratch_dir = make_fresh_dir('background')
scratch_dir


# ### Make an observation table defining the run grouping
# 
# Prepare a scheme to group observations with similar observing conditions and create a new ObservationTable with the grouping ID for each run

# In[6]:


# Create a background model from the 4 Crab runs for the counts ouside the exclusion region so here outside the Crab
data_store = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2")

# Define the grouping you want to use to group the obervations to make the acceptance curves
# Here we use 2 Zenith angle bins only, you can also add efficiency bins for example etc...
axes = [ObservationGroupAxis('ZEN_PNT', [0, 49, 90], fmt='edges')]

# Create the ObservationGroups object
obs_groups = ObservationGroups(axes)
# write it to file
filename = str(scratch_dir / 'group-def.fits')
obs_groups.obs_groups_table.write(filename, overwrite=True)

# Create a new ObservationTable with the column group_id
# You give the runs list you want to use to produce the background model that are in your obs table. 
# Here very simple only the 4 Crab runs...
list_ids = [23523, 23526, 23559, 23592]
obs_table_with_group_id = obs_groups.apply(data_store.obs_table.select_obs_id(list_ids))


# ### Make table of known gamma-ray sources to exclude
# 
# We need a mask to remove known sources from the observation. We use TeVcat and exclude a circular region of at least 0.3° radius. Here since we use Crab runs, we will remove the Crab events from the FOV to select only the OFF events to build the acceptance curves. Of cource normally you use thousand of AGN runs to build coherent acceptance curves.

# In[7]:


cat = SourceCatalogGammaCat()
exclusion_table = cat.table.copy()
exclusion_table.rename_column('ra', 'RA')
exclusion_table.rename_column('dec', 'DEC')
radius = exclusion_table['morph_sigma'].data
radius[np.isnan(radius)] = 0.3
exclusion_table['Radius'] = radius * u.deg
exclusion_table = Table(exclusion_table)


# ### Run the OffDataBackgroundMaker
# 
# Make the acceptance curves in the different group of observation conditions you defined above using the obs_table containaing the group id for each observation used to compute the bkg model

# In[8]:


bgmaker = OffDataBackgroundMaker(
    data_store=data_store,
    outdir=str(scratch_dir),
    run_list=None,
    obs_table=obs_table_with_group_id,
    ntot_group=obs_groups.n_groups,
    excluded_sources=exclusion_table,
)

# Define the energy and offset binning to use
ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 15, 'TeV')
offset = sqrt_space(start=0, stop=2.5, num=100) * u.deg

# Make the model (i.e. stack counts and livetime)
bgmaker.make_model("2D", ebounds=ebounds, offset=offset)

# Smooth the model
bgmaker.smooth_models("2D")

# Write the model to disk
bgmaker.save_models("2D")
bgmaker.save_models(modeltype="2D", smooth=True)


# Congratulations, you have produced a background model.
# 
# The following files were generated in our scratch directory:

# In[9]:


[path.name for path in scratch_dir.glob('*')]


# ## Inspect the background model
# 
# Our template background model has two axes: offset and energy.
# 
# Let's make a few plots to see what it looks like:
# 1. Acceptance curve (background rate as a function of field of view offset for a given energy)
# 1. Rate spectrum (background rate as a function of energy for a given offset)
# 1. Rate image (background rate as a function of energy and offset)
# 
# ### Acceptance curve

# In[10]:


# Read one of the background models from file
filename = scratch_dir / 'smooth_background_2D_group_000_table.fits.gz'
model = EnergyOffsetBackgroundModel.read(str(filename))


# In[11]:


offset = model.bg_rate.offset_bin_center
energies = model.bg_rate.energy  
iE = 6

x = offset
y = model.bg_rate.data[iE,:]
plt.plot(x, y, label="bkg model smooth")
title = "energy band: "+str("%.2f"%energies[iE].value)+"-"+str("%.2f"%energies[iE+1].value)+" TeV"
plt.title(title)
plt.xlabel("Offset (degree)")
plt.ylabel("Bkg rate (MeV-1 s-1 sr-1)")
plt.legend()


# ### Background rate spectrum

# In[12]:


x = energies.log_centers
y = model.bg_rate.data[:,10]
plt.loglog(x, y, label="bkg model smooth")
plt.title("offset: "+str("%.2f"%offset[10].value)+" deg")
plt.xlabel("Energy (TeV)")
plt.ylabel("Bkg rate (MeV-1 s-1 sr-1)")


# ### Background rate image with energy and offset axes
# 
# It doesn't look good in this case.
# To do this well, you need to use more off or AGN runs to build the background model!

# In[13]:


model.bg_rate.plot()


# ## Make new HDU index table including background
# 
# Here we first copy the dataset of the 4 crab runs from gammapy-extra in a new directory containing the data you will use for the analysis. 
# 
# We use the same dataset to produce the bkg or for the analysis. Of course normally you produce the bkg model using thousands of AGN runs not the 4 Crab test runs.

# In[14]:


# Make a new hdu table in your dataset directory that contains the link to the acceptance curve to use to build the bkg model in your cube analysis
data_dir = make_fresh_dir('data')


# In[15]:


ds = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2")
ds.copy_obs(ds.obs_table, data_dir)


# The hdu_table in this directory contains no link to a bkg model for each observation. 

# In[16]:


data_store = DataStore.from_dir(data_dir)
data_store.hdu_table


# In order to produce a background image or background cube we have to create a hdu table that contains for each observation a link to the bkg model to use depending of the observation conditions of the run.

# In[17]:


#Copy the background directory in the one where is located the hdu table, here data
shutil.move(str(scratch_dir), str(data_dir))

# Create the new hdu table with a link to the background model
group_filename = data_dir / 'background/group-def.fits'

#relat_path= (scratch_dir.absolute()).relative_to(data_dir.absolute())
hdu_index_table = bgmaker.make_total_index_table(
    data_store=data_store,
    modeltype='2D',
    out_dir_background_model=scratch_dir,
    filename_obs_group_table=str(group_filename),
    smooth=False,
)

# Write the new hdu table
filename = data_dir / 'hdu-index.fits.gz'
hdu_index_table.write(str(filename), overwrite=True)


# In[18]:


hdu_index_table


# In[19]:


print(hdu_index_table[-4]["FILE_DIR"], " ", hdu_index_table[-4]["FILE_NAME"])


# In[20]:


print(hdu_index_table[-1]["FILE_DIR"], " ", hdu_index_table[-1]["FILE_NAME"])


# ## Exercises
# 
# - Use real AGN run
# - Change the binning for the grouping: thinner zenithal bin, add efficiency binning ....
# - Change the energy binning (ebounds) and the offset (offset) used to compute the acceptance curve
# 

# ## What next?
# 
# In this tutorial we have created a template background model in the `bkg_2d` format, i.e. with offset and energy axes (see [spec](http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/background/index.html#bkg-2d-format)).
# 
# In future tutorials, we will use this background model as one of the model components for source analysis.
