
# coding: utf-8

# # Spectrum simulation for CTA
# 
# A quick example how to simulate and fit a spectrum for the [Cherenkov Telescope Array (CTA)](https://www.cta-observatory.org).
# 
# We will use the following classes:
# 
# * [gammapy.spectrum.SpectrumObservation](http://docs.gammapy.org/0.8/api/gammapy.spectrum.SpectrumObservation.html)
# * [gammapy.spectrum.SpectrumSimulation](http://docs.gammapy.org/0.8/api/gammapy.spectrum.SpectrumSimulation.html)
# * [gammapy.spectrum.SpectrumFit](http://docs.gammapy.org/0.8/api/gammapy.spectrum.SpectrumFit.html)
# * [gammapy.irf.CTAIrf](http://docs.gammapy.org/0.8/api/gammapy.irf.CTAIrf.html)

# ## Setup

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


import numpy as np
import astropy.units as u
from gammapy.irf import EnergyDispersion, EffectiveAreaTable
from gammapy.spectrum import SpectrumSimulation, SpectrumFit
from gammapy.spectrum.models import PowerLaw
from gammapy.irf import CTAIrf


# ## Simulation

# In[ ]:


# Define simulation parameters parameters
livetime = 1 * u.h
offset = 0.5 * u.deg
# Energy from 0.1 to 100 TeV with 10 bins/decade
energy = np.logspace(-1, 2, 31) * u.TeV


# In[ ]:


# Define spectral model
model = PowerLaw(
    index=2.1,
    amplitude=2.5e-12 * u.Unit("cm-2 s-1 TeV-1"),
    reference=1 * u.TeV,
)


# In[ ]:


# Load IRFs
filename = (
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)
cta_irf = CTAIrf.read(filename)


# In[ ]:


aeff = cta_irf.aeff.to_effective_area_table(offset=offset, energy=energy)
aeff.plot()
plt.loglog()
print(cta_irf.aeff.data)


# In[ ]:


edisp = cta_irf.edisp.to_energy_dispersion(
    offset=offset, e_true=energy, e_reco=energy
)
edisp.plot_matrix()
print(edisp.data)


# In[ ]:


# Simulate data
sim = SpectrumSimulation(
    aeff=aeff, edisp=edisp, source_model=model, livetime=livetime
)
sim.simulate_obs(seed=42, obs_id=0)


# In[ ]:


sim.obs.peek()
print(sim.obs)


# ## Spectral analysis
# 
# Now that we have some simulated CTA counts spectrum, let's analyse it.

# In[ ]:


# Fit data
fit = SpectrumFit(obs_list=sim.obs, model=model, stat="cash")
fit.run()
result = fit.result[0]


# In[ ]:


print(result)


# In[ ]:


energy_range = [0.1, 100] * u.TeV
model.plot(energy_range=energy_range, energy_power=2)
result.model.plot(energy_range=energy_range, energy_power=2)
result.model.plot_error(energy_range=energy_range, energy_power=2);


# ## Exercises
# 
# * Change the observation time to something longer or shorter. Does the observation and spectrum results change as you expected?
# * Change the spectral model, e.g. add a cutoff at 5 TeV, or put a steep-spectrum source with spectral index of 4.0

# In[ ]:


# Start the exercises here!


# ## What next?
# 
# In this tutorial we simulated and analysed the spectrum of source using CTA prod 2 IRFs.
# 
# If you'd like to go further, please see the other tutorial notebooks.
