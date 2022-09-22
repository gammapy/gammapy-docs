
# coding: utf-8

# # Flux point fitting in Gammapy
# 
# 
# ## Introduction
# 
# In this tutorial we're going to learn how to fit spectral models to combined Fermi-LAT and IACT flux points.
# 
# The central class we're going to use for this example analysis is:  
# 
# - [gammapy.spectrum.FluxPointFit](https://docs.gammapy.org/0.9/api/gammapy.spectrum.FluxPointFit.html)
# 
# In addition we will work with the following data classes:
# 
# - [gammapy.spectrum.FluxPoints](https://docs.gammapy.org/0.9/api/gammapy.spectrum.FluxPoints.html)
# - [gammapy.catalog.SourceCatalogGammaCat](https://docs.gammapy.org/0.9/api/gammapy.catalog.SourceCatalogGammaCat.html)
# - [gammapy.catalog.SourceCatalog3FHL](https://docs.gammapy.org/0.9/api/gammapy.catalog.SourceCatalog3FHL.html)
# - [gammapy.catalog.SourceCatalog3FGL](https://docs.gammapy.org/0.9/api/gammapy.catalog.SourceCatalog3FGL.html)
# 
# And the following spectral model classes:
# 
# - [PowerLaw](https://docs.gammapy.org/0.9/api/gammapy.spectrum.models.PowerLaw.html)
# - [ExponentialCutoffPowerLaw](https://docs.gammapy.org/0.9/api/gammapy.spectrum.models.ExponentialCutoffPowerLaw.html)
# - [LogParabola](https://docs.gammapy.org/0.9/api/gammapy.spectrum.models.LogParabola.html)

# ## Setup
# 
# Let us start with the usual IPython notebook and Python imports:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


from astropy import units as u
from gammapy.spectrum.models import (
    PowerLaw,
    ExponentialCutoffPowerLaw,
    LogParabola,
)
from gammapy.spectrum import FluxPointFit, FluxPoints
from gammapy.catalog import (
    SourceCatalog3FGL,
    SourceCatalogGammaCat,
    SourceCatalog3FHL,
)


# ## Load spectral points
# 
# For this analysis we choose to work with the source 'HESS J1507-622' and the associated Fermi-LAT sources '3FGL J1506.6-6219' and '3FHL J1507.9-6228e'. We load the source catalogs, and then access source of interest by name:

# In[ ]:


fermi_3fgl = SourceCatalog3FGL()
fermi_3fhl = SourceCatalog3FHL()
gammacat = SourceCatalogGammaCat()


# In[ ]:


source_gammacat = gammacat["HESS J1507-622"]
source_fermi_3fgl = fermi_3fgl["3FGL J1506.6-6219"]
source_fermi_3fhl = fermi_3fhl["3FHL J1507.9-6228e"]


# The corresponding flux points data can be accessed with `.flux_points` attribute:

# In[ ]:


flux_points_gammacat = source_gammacat.flux_points
flux_points_gammacat.table


# In the Fermi-LAT catalogs, integral flux points are given. Currently the flux point fitter only works with differential flux points, so we apply the conversion here.

# In[ ]:


flux_points_3fgl = source_fermi_3fgl.flux_points.to_sed_type(
    sed_type="dnde", model=source_fermi_3fgl.spectral_model
)
flux_points_3fhl = source_fermi_3fhl.flux_points.to_sed_type(
    sed_type="dnde", model=source_fermi_3fhl.spectral_model
)


# Finally we stack the flux points into a single `FluxPoints` object and drop the upper limit values, because currently we can't handle them in the fit:

# In[ ]:


# stack flux point tables
flux_points = FluxPoints.stack(
    [flux_points_gammacat, flux_points_3fhl, flux_points_3fgl]
)

# drop the flux upper limit values
flux_points = flux_points.drop_ul()


# ## Power Law Fit
# 
# First we start with fitting a simple [power law](https://docs.gammapy.org/0.9/api/gammapy.spectrum.models.PowerLaw.html#gammapy.spectrum.models.PowerLaw).

# In[ ]:


pwl = PowerLaw(index=2, amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV")


# After creating the model we run the fit by passing the `'flux_points'` and `'pwl'` objects:

# In[ ]:


fitter = FluxPointFit(pwl, flux_points, stat="chi2assym")
result_pwl = fitter.run()


# And print the result:

# In[ ]:


print(result_pwl.model)


# Finally we plot the data points and the best fit model:

# In[ ]:


ax = flux_points.plot(energy_power=2)
result_pwl.model.plot(energy_range=[1e-4, 1e2] * u.TeV, ax=ax, energy_power=2)
result_pwl.model.plot_error(
    energy_range=[1e-4, 1e2] * u.TeV, ax=ax, energy_power=2
)
ax.set_ylim(1e-13, 1e-11);


# ## Exponential Cut-Off Powerlaw Fit
# 
# Next we fit an [exponential cut-off power](https://docs.gammapy.org/0.9/api/gammapy.spectrum.models.ExponentialCutoffPowerLaw.html#gammapy.spectrum.models.ExponentialCutoffPowerLaw) law to the data.

# In[ ]:


ecpl = ExponentialCutoffPowerLaw(
    index=2,
    amplitude="1e-12 cm-2 s-1 TeV-1",
    reference="1 TeV",
    lambda_="0.1 TeV-1",
)


# We run the fitter again by passing the flux points and the `ecpl` model instance:

# In[ ]:


fitter = FluxPointFit(ecpl, flux_points, stat="chi2assym")
result_ecpl = fitter.run()
print(result_ecpl.model)


# We plot the data and best fit model:

# In[ ]:


ax = flux_points.plot(energy_power=2)
result_ecpl.model.plot(energy_range=[1e-4, 1e2] * u.TeV, ax=ax, energy_power=2)
result_ecpl.model.plot_error(
    energy_range=[1e-4, 1e2] * u.TeV, ax=ax, energy_power=2
)
ax.set_ylim(1e-13, 1e-11)


# ## Log-Parabola Fit
# 
# Finally we try to fit a [log-parabola](https://docs.gammapy.org/0.9/api/gammapy.spectrum.models.LogParabola.html#gammapy.spectrum.models.LogParabola) model:

# In[ ]:


log_parabola = LogParabola(
    alpha=2, amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV", beta=0.1
)


# In[ ]:


fitter = FluxPointFit(log_parabola, flux_points, stat="chi2assym")
result_log_parabola = fitter.run()
print(result_log_parabola.model)


# In[ ]:


ax = flux_points.plot(energy_power=2)
result_log_parabola.model.plot(
    energy_range=[1e-4, 1e2] * u.TeV, ax=ax, energy_power=2
)
result_log_parabola.model.plot_error(
    energy_range=[1e-4, 1e2] * u.TeV, ax=ax, energy_power=2
)
ax.set_ylim(1e-13, 1e-11);


# ## Exercises
# 
# - Fit a `PowerLaw2` and `ExponentialCutoffPowerLaw3FGL` to the same data.
# - Fit a `ExponentialCutoffPowerLaw` model to Vela X ('HESS J0835-455') only and check if the best fit values correspond to the values given in the Gammacat catalog

# ## What next?
# 
# This was an introduction to SED fitting in Gammapy.
# 
# * If you would like to learn how to perform a full Poisson maximum likelihood spectral fit, please check out the [spectrum pipe](spectrum_pipe.ipynb) tutorial.
# * If you are interested in simulation of spectral data in the context of CTA, please check out the [spectrum simulation cta](spectrum_simulation_cta.ipynb) notebook.
# * To learn more about other parts of Gammapy (e.g. Fermi-LAT and TeV data analysis), check out the other tutorial notebooks.
# * To see what's available in Gammapy, browse the [Gammapy docs](https://docs.gammapy.org/) or use the full-text search.
# * If you have any questions, ask on the mailing list .
