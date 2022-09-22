
# coding: utf-8

# # Computation of the CTA sensitivity

# ## Introduction
# 
# This notebook explains how to derive the CTA sensitivity for a point-like IRF at a fixed zenith angle and fixed offset. The significativity is computed for the 1D analysis (On-OFF regions) and the LiMa formula.
# 
# We will be using the following Gammapy classes:
# 
# * gammapy.irf.CTAIrf
# * [gammapy.spectrum.SensitivityEstimator](https://docs.gammapy.org/0.9/api/gammapy.spectrum.SensitivityEstimator.html)

# ## Setup
# As usual, we'll start with some setup ...

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


from gammapy.irf import CTAPerf
from gammapy.spectrum import SensitivityEstimator


# ## Load IRFs
# 
# First load the CTA IRFs.

# In[ ]:


filename = "$GAMMAPY_EXTRA/datasets/cta/perf_prod2/point_like_non_smoothed/South_5h.fits.gz"
irf = CTAPerf.read(filename)


# ## Compute sensitivity
# 
# Choose a few parameters, then run the sentitivity computation.

# In[ ]:


sensitivity_estimator = SensitivityEstimator(irf=irf, livetime="5h")
sensitivity_estimator.run()


# ## Results
# 
# The results are given as an Astropy table.

# In[ ]:


# Show the results table
sensitivity_estimator.results_table


# In[ ]:


# Save it to file (could use e.g. format of CSV or ECSV or FITS)
# sensitivity_estimator.results_table.write('sensitivity.ecsv', format='ascii.ecsv')


# In[ ]:


# Plot the sensitivity curve
t = sensitivity_estimator.results_table

is_s = t["criterion"] == "significance"
plt.plot(
    t["energy"][is_s],
    t["e2dnde"][is_s],
    "s-",
    color="red",
    label="significance",
)

is_g = t["criterion"] == "gamma"
plt.plot(
    t["energy"][is_g], t["e2dnde"][is_g], "*-", color="blue", label="gamma"
)

plt.loglog()
plt.xlabel("Energy ({})".format(t["energy"].unit))
plt.ylabel("Sensitivity ({})".format(t["e2dnde"].unit))
plt.legend();


# ## Exercises
# 
# * Also compute the sensitivity for a 20 hour observation
# * Compare how the sensitivity differs between 5 and 20 hours by plotting the ratio as a function of energy.
