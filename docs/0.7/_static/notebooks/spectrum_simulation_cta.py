
# coding: utf-8

# # Spectrum simulation for CTA
# 
# A quick example how to simulate and fit a spectrum for the [Cherenkov Telescope Array (CTA)](https://www.cta-observatory.org)
# 
# We will use the following classes:
# 
# * [gammapy.spectrum.SpectrumObservation](http://docs.gammapy.org/0.7/api/gammapy.spectrum.SpectrumObservation.html)
# * [gammapy.spectrum.SpectrumSimulation](http://docs.gammapy.org/0.7/api/gammapy.spectrum.SpectrumSimulation.html)
# * [gammapy.spectrum.SpectrumFit](http://docs.gammapy.org/0.7/api/gammapy.spectrum.SpectrumFit.html)
# * [gammapy.scripts.CTAIrf](http://docs.gammapy.org/0.7/api/gammapy.scripts.CTAIrf.html)

# ## Setup

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


import numpy as np
import astropy.units as u
from gammapy.irf import EnergyDispersion, EffectiveAreaTable
from gammapy.spectrum import SpectrumSimulation, SpectrumFit
from gammapy.spectrum.models import PowerLaw
from gammapy.scripts import CTAIrf


# ## Simulation

# In[3]:


# Define obs parameters
livetime = 10 * u.min
offset = 0.3 * u.deg
lo_threshold = 0.1 * u.TeV
hi_threshold = 60 * u.TeV


# In[4]:


# Define spectral model
index = 2.3 * u.Unit('')
amplitude = 2.5 * 1e-12 * u.Unit('cm-2 s-1 TeV-1')
reference = 1 * u.TeV
model = PowerLaw(index=index, amplitude=amplitude, reference=reference)


# In[5]:


# Load IRFs
filename = '$GAMMAPY_EXTRA/datasets/cta/perf_prod2/South_5h/irf_file.fits.gz'
cta_irf = CTAIrf.read(filename)


# In[6]:


aeff = cta_irf.aeff.to_effective_area_table(offset=offset)
aeff.plot()
print(cta_irf.aeff.data)


# In[7]:


edisp = cta_irf.edisp.to_energy_dispersion(offset=offset)
edisp.plot_matrix()
print(edisp.data)


# In[8]:


# Simulate data
aeff.lo_threshold = lo_threshold
aeff.hi_threshold = hi_threshold
sim = SpectrumSimulation(aeff=aeff, edisp=edisp, source_model=model, livetime=livetime)
sim.simulate_obs(seed=42, obs_id=0)


# In[9]:


sim.obs.peek()
print(sim.obs)


# ## Spectral analysis
# 
# Now that we have some simulated CTA counts spectrum, let's analyse it.

# In[10]:


# Fit data
fit = SpectrumFit(obs_list=sim.obs, model=model, stat='cash')
fit.run()
result = fit.result[0]


# In[11]:


print(result)


# In[12]:


energy_range = [0.1, 100] * u.TeV
model.plot(energy_range=energy_range, energy_power=2)
result.model.plot(energy_range=energy_range, energy_power=2)
result.model.plot_error(energy_range=energy_range, energy_power=2)


# ## Exercises
# 
# * Change the observation time to something longer or shorter. Does the observation and spectrum results change as you expected?
# * Change the spectral model, e.g. add a cutoff at 5 TeV, or put a steep-spectrum source with spectral index of 4.0

# In[13]:


# Start the exercises here!


# ## What next?
# 
# In this tutorial we simulated and analysed the spectrum of source using CTA prod 2 IRFs.
# 
# If you'd like to go further, please see the other tutorial notebooks.
