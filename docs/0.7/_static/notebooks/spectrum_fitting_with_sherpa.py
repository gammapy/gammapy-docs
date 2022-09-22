
# coding: utf-8

# # Fitting gammapy spectra with sherpa
# 
# Once we have exported the spectral files (PHA, ARF, RMF and BKG) in the OGIP format, it becomes possible to fit them later with gammapy or with any existing OGIP compliant tool such as XSpec or sherpa.
# 
# We show here how to do so with sherpa using the high-level user interface. For a general view on how to use stand-alone sherpa, see this [tutorial](http://nbviewer.jupyter.org/github/sherpa/sherpa/blob/master/docs/SherpaQuickStart.ipynb)
# 
# Tu run this notebook, one should first run spectrum_analysis to extract the relevant OGIP files.

# ## Importing sherpa high level UI
# 
# ### loading the data as a sherpa DataStack
# 
# We first need to import the user interface and load the data with [load_data](http://cxc.harvard.edu/sherpa/ahelp/load_data.html). One can load files one by one, or more simply load them all at once through a [DataStack](http://cxc.harvard.edu/sherpa/ahelp/datastack.html).

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import glob   # to list files
from sherpa.astro.datastack import DataStack
import sherpa.astro.datastack as sh


# In[2]:


ANALYSIS_DIR = 'crab_analysis'

ds = DataStack()
pha_list = glob.glob(ANALYSIS_DIR + '/ogip_data/pha_obs*.fits')
for pha in pha_list:
    sh.load_data(ds,pha)
    
# see what is stored
ds.show_stack()


# ## Creating the source model
# 
# We can now use sherpa models. We need to remember that they were designed for X-ray astronomy and energy is written in keV. 
# 
# Here we start with a simple PL.

# In[3]:


# Define the source model
ds.set_source("powlaw1d.p1")

# Change reference energy of the model 
p1.ref = 1e9   # 1 TeV = 1e9 keV
p1.gamma = 2.0
p1.ampl = 1e-20  # in cm**-2 s**-1 keV**-1
# View parameters 
print(p1)


# ## Perform the fit and compute errors
# 
# We need to set the correct statistic: [WSTAT](http://cxc.harvard.edu/sherpa/ahelp/wstat.html). We use functions [set_stat](http://cxc.harvard.edu/sherpa/ahelp/set_stat.html) to define the fit statistic, [notice](http://cxc.harvard.edu/sherpa/ahelp/notice.html) to set the energy range, and [fit](http://cxc.harvard.edu/sherpa/ahelp/fit.html)

# In[4]:


### Define the statistic
sh.set_stat("wstat")

### Define the fit range
ds.notice(0.6e9,20e9)

### Do the fit
ds.fit()


# ## Plot the result
# 
# Note that sherpa does not provide flux points. It also only provides plot for each individual spectrum.

# In[5]:


sh.get_data_plot_prefs()["xlog"] = True
sh.get_data_plot_prefs()["ylog"] = True
ds.plot_fit()


# ## Errors and confidence contours
# 
# We use [conf](http://cxc.harvard.edu/sherpa/ahelp/conf.html) and [reg_proj](http://cxc.harvard.edu/sherpa/ahelp/reg_proj.html) functions.

# In[6]:


### Compute confidnce intervals
ds.conf()


# In[7]:


### Compute confidence contours for amplitude and index
sh.reg_unc(p1.gamma,p1.ampl)

