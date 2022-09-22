
# coding: utf-8

# # Computation of the CTA sensitivity

# ## Introduction
# 
# This notebook explains how to derive the CTA sensitivity for a point-like IRF at a fixed zenith angle and fixed offset. The significativity is computed for the 1D analysis (On-OFF regions) and the LiMa formula.
# 
# We will be using the following Gammapy classes:
# 
# * [gammapy.scripts.CTAIrf](http://docs.gammapy.org/0.7/api/gammapy.scripts.CTAIrf.html)
# * [gammapy.scripts.SensitivityEstimator](http://docs.gammapy.org/0.7/api/gammapy.scripts.SensitivityEstimator.html)

# ## Setup
# As usual, we'll start with some setup ...

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from gammapy.scripts import CTAPerf, SensitivityEstimator


# ## Load IRFs
# 
# First import the CTA IRFs

# In[3]:


filename = '$GAMMAPY_EXTRA/datasets/cta/perf_prod2/point_like_non_smoothed/South_5h.fits.gz'
irf = CTAPerf.read(filename)


# ## Compute sensitivity
# 
# Choose a few parameters, then run the sentitivity computation.

# In[4]:


sens = SensitivityEstimator(
    irf=irf,
    livetime='5h',
)
sens.run()


# ## Print and plot the results

# In[5]:


sens.print_results()


# In[6]:


sens.plot()


# In[7]:


# This will give you the results as an Astropy table,
# which you can save to FITS or CSV or use for further analysis
sens.diff_sensi_table


# ## Exercises
# 
# * tbd
