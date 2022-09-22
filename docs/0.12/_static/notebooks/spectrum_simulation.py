
# coding: utf-8

# # Spectrum simulation for CTA
# 
# A quick example how to use the functions and classes in gammapy.spectrum in order to simulate and fit spectra. 
# 
# We will simulate observations for the [Cherenkov Telescope Array (CTA)](https://www.cta-observatory.org) first using a power law model without any background. Than we will add a power law shaped background component. The next part of the tutorial shows how to use user defined models for simulations and fitting.
# 
# We will use the following classes:
# 
# * [gammapy.spectrum.SpectrumDatasetOnOff](https://docs.gammapy.org/0.12/api/gammapy.spectrum.SpectrumDatasetOnOff.html)
# * [gammapy.spectrum.SpectrumSimulation](https://docs.gammapy.org/0.12/api/gammapy.spectrum.SpectrumSimulation.html)
# * [gammapy.irf.load_cta_irfs](https://docs.gammapy.org/0.12/api/gammapy.irf.load_cta_irfs.html)
# * [gammapy.spectrum.models.PowerLaw](https://docs.gammapy.org/0.12/api/gammapy.spectrum.models.PowerLaw.html)

# ## Setup
# 
# Same procedure as in every script ...

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


import numpy as np
import astropy.units as u
from gammapy.spectrum import SpectrumSimulation
from gammapy.utils.fitting import Fit, Parameter
from gammapy.spectrum.models import PowerLaw
from gammapy.spectrum import models
from gammapy.irf import load_cta_irfs


# ## Simulation of a single spectrum
# 
# To do a simulation, we need to define the observational parameters like the livetime, the offset, the energy range to perform the simulation for and the choice of spectral model. This will then be convolved with the IRFs, and Poission fluctuated, to get the simulated counts for each observation.  

# In[ ]:


# Define simulation parameters parameters
livetime = 1 * u.h
offset = 0.5 * u.deg
# Energy from 0.1 to 100 TeV with 10 bins/decade
energy = np.logspace(-1, 2, 31) * u.TeV


# In[ ]:


# Define spectral model - a simple Power Law in this case
model_ref = PowerLaw(
    index=3.0,
    amplitude=2.5e-12 * u.Unit("cm-2 s-1 TeV-1"),
    reference=1 * u.TeV,
)
print(model_ref)


# ### Get and set the model parameters after initialising
# The model parameters are stored in the `Parameters` object on the spectal model. Each model parameter is a `Parameter` instance. It has a `value` and a `unit` attribute, as well as a `quantity` property for convenience.

# In[ ]:


print(model_ref.parameters)


# In[ ]:


print(model_ref.parameters["index"])
model_ref.parameters["index"].value = 2.1
print(model_ref.parameters["index"])


# In[ ]:


# Load IRFs
filename = (
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)
cta_irf = load_cta_irfs(filename)


# A quick look into the effective area and energy dispersion:

# In[ ]:


aeff = cta_irf["aeff"].to_effective_area_table(offset=offset, energy=energy)
aeff.plot()
plt.loglog()
print(cta_irf["aeff"].data)


# In[ ]:


edisp = cta_irf["edisp"].to_energy_dispersion(
    offset=offset, e_true=energy, e_reco=energy
)
edisp.plot_matrix()
print(edisp.data)


# The `SpectrumSimulation` class does the work of convolving the model with the effective area and the energy dispersion, and then Poission fluctuating the counts. An `obs_id` is needed by `SpectrumSimulation.simulate_obs()` to keep track of the simulated spectra. Here, we just pass a dummy index, but while simulating observations in a loop, this needs to be updated.

# In[ ]:


# Simulate data
sim = SpectrumSimulation(
    aeff=aeff, edisp=edisp, source_model=model_ref, livetime=livetime
)
sim.simulate_obs(seed=42, obs_id=0)


# In[ ]:


# Take a quick look at the simulated counts
sim.obs.peek()
print(sim.obs)


# ## Include Background 
# 
# In this section we will include a background component. Furthermore, we will also simulate more than one observation and fit each one individually in order to get average fit results.

# In[ ]:


# We assume a PowerLaw shape of the background as well
bkg_model = PowerLaw(
    index=2.5, amplitude=1e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Now simulate 30 indepenent spectra using the same set of observation conditions.\nn_obs = 30\nseeds = np.arange(n_obs)\n\nsim = SpectrumSimulation(\n    aeff=aeff,\n    edisp=edisp,\n    source_model=model_ref,\n    livetime=livetime,\n    background_model=bkg_model,\n    alpha=0.2,\n)\n\nsim.run(seeds)\nprint(sim.result)\nprint(sim.result[0])')


# Before moving on to the fit let's have a look at the simulated observations.

# In[ ]:


n_on = [obs.total_stats.n_on for obs in sim.result]
n_off = [obs.total_stats.n_off for obs in sim.result]
excess = [obs.total_stats.excess for obs in sim.result]

fix, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].hist(n_on)
axes[0].set_xlabel("n_on")
axes[1].hist(n_off)
axes[1].set_xlabel("n_off")
axes[2].hist(excess)
axes[2].set_xlabel("excess");


# Now, we fit each simulated spectrum individually 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'results = []\nfor obs in sim.result:\n    dataset = obs\n    dataset.model = model_ref.copy()\n    fit = Fit([dataset])\n    result = fit.optimize()\n    results.append(\n        {\n            "index": result.parameters["index"].value,\n            "amplitude": result.parameters["amplitude"].value,\n        }\n    )')


# We take a look at the distribution of the fitted indices. This matches very well with the spectrum that we initially injected, index=2.1

# In[ ]:


index = np.array([_["index"] for _ in results])
plt.hist(index, bins=10, alpha=0.5)
plt.axvline(x=model_ref.parameters["index"].value, color="red")
print("spectral index: {:.2f} +/- {:.2f}".format(index.mean(), index.std()))


# ## Adding a user defined model
# 
# Many spectral models in gammapy are subclasses of `SpectralModel`. The list of available models is shown below.

# In[ ]:


models.SpectralModel.__subclasses__()


# This section shows how to add a user defined spectral model. 
# 
# To do that you need to subclass `SpectralModel`. All `SpectralModel` subclasses need to have an `__init__` function, which sets up the `Parameters` of the model and a `static` function called `evaluate` where the mathematical expression for the model is defined.
# 
# As an example we will use a PowerLaw plus a Gaussian (with fixed width).

# In[ ]:


class UserModel(models.SpectralModel):
    def __init__(self, index, amplitude, reference, mean, width):
        super().__init__(
            [
                Parameter("index", index, min=0),
                Parameter("amplitude", amplitude, min=0),
                Parameter("reference", reference, frozen=True),
                Parameter("mean", mean, min=0),
                Parameter("width", width, min=0, frozen=True),
            ]
        )

    @staticmethod
    def evaluate(energy, index, amplitude, reference, mean, width):
        pwl = models.PowerLaw.evaluate(
            energy=energy,
            index=index,
            amplitude=amplitude,
            reference=reference,
        )
        gauss = amplitude * np.exp(-(energy - mean) ** 2 / (2 * width ** 2))
        return pwl + gauss


# In[ ]:


model = UserModel(
    index=2,
    amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
    reference=1 * u.TeV,
    mean=5 * u.TeV,
    width=0.2 * u.TeV,
)
print(model)


# In[ ]:


energy_range = [1, 10] * u.TeV
model.plot(energy_range=energy_range);


# ## Exercises
# 
# * Change the observation time to something longer or shorter. Does the observation and spectrum results change as you expected?
# * Change the spectral model, e.g. add a cutoff at 5 TeV, or put a steep-spectrum source with spectral index of 4.0
# * Simulate spectra with the spectral model we just defined. How much observation duration do you need to get back the injected parameters?

# ## What next?
# 
# In this tutorial we simulated and analysed the spectrum of source using CTA prod 2 IRFs.
# 
# If you'd like to go further, please see the other tutorial notebooks.
