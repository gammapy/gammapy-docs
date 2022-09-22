
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
# - [gammapy.spectrum.FluxPointFitter](http://docs.gammapy.org/0.7/api/gammapy.spectrum.FluxPointFitter.html)
# 
# In addition we will work with the following data classes:
# 
# - [gammapy.spectrum.FluxPoints](http://docs.gammapy.org/0.7/api/gammapy.spectrum.FluxPoints.html)
# - [gammapy.catalog.SourceCatalogGammaCat](http://docs.gammapy.org/0.7/api/gammapy.catalog.SourceCatalogGammaCat.html)
# - [gammapy.catalog.SourceCatalog3FHL](http://docs.gammapy.org/0.7/api/gammapy.catalog.SourceCatalog3FHL.html)
# - [gammapy.catalog.SourceCatalog3FGL](http://docs.gammapy.org/0.7/api/gammapy.catalog.SourceCatalog3FGL.html)
# 
# And the following spectral model classes:
# 
# - [PowerLaw](http://docs.gammapy.org/0.7/api/gammapy.spectrum.models.PowerLaw.html)
# - [ExponentialCutoffPowerLaw](http://docs.gammapy.org/0.7/api/gammapy.spectrum.models.ExponentialCutoffPowerLaw.html)
# - [LogParabola](http://docs.gammapy.org/0.7/api/gammapy.spectrum.models.LogParabola.html)

# ## Setup
# 
# Let us start with the usual IPython notebook and Python imports:

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from astropy import units as u
from astropy.table import vstack
from gammapy.spectrum.models import PowerLaw, ExponentialCutoffPowerLaw, LogParabola
from gammapy.spectrum import FluxPointFitter, FluxPoints
from gammapy.catalog import SourceCatalog3FGL, SourceCatalogGammaCat, SourceCatalog3FHL


# ## Load spectral points
# 
# For this analysis we choose to work with the source 'HESS J1507-622' and the associated Fermi-LAT sources '3FGL J1506.6-6219' and '3FHL J1507.9-6228e'. We load the source catalogs, and then access source of interest by name:

# In[3]:


fermi_3fgl = SourceCatalog3FGL()
fermi_3fhl = SourceCatalog3FHL()
gammacat = SourceCatalogGammaCat()


# In[4]:


source_gammacat =  gammacat['HESS J1507-622']
source_fermi_3fgl = fermi_3fgl['3FGL J1506.6-6219']
source_fermi_3fhl = fermi_3fhl['3FHL J1507.9-6228e']


# The corresponding flux points data can be accessed with `.flux_points` attribute:

# In[5]:


flux_points_gammacat = source_gammacat.flux_points
flux_points_gammacat.table


# In the Fermi-LAT catalogs, integral flux points are given. Currently the flux point fitter only works with differential flux points, so we apply the conversion here.

# In[6]:


flux_points_3fgl = source_fermi_3fgl.flux_points.to_sed_type(
    sed_type='dnde',
    model=source_fermi_3fgl.spectral_model,
)
flux_points_3fhl = source_fermi_3fhl.flux_points.to_sed_type(
    sed_type='dnde',
    model=source_fermi_3fhl.spectral_model,
)


# Finally we stack the flux points into a single `FluxPoints` object and drop the upper limit values, because currently we can't handle them in the fit:

# In[7]:


# stack flux point tables
flux_points = FluxPoints.stack([
    flux_points_gammacat,
    flux_points_3fhl,
    flux_points_3fgl
])

# drop the flux upper limit values
flux_points = flux_points.drop_ul()


# ## Fitter Setup

# We initialze the fitter object with the `'chi2assym'` statistic, because we have assymmetric errors on the flux points. As optimizer we choose the `'simplex'` algorithm and to estimate the errors we use `'covar'` method: 

# In[8]:


fitter = FluxPointFitter(
    stat='chi2assym',
    optimizer='simplex',
    error_estimator='covar',
)


# ## Power Law Fit
# 
# First we start with fitting a simple [power law](http://docs.gammapy.org/0.7/api/gammapy.spectrum.models.PowerLaw.html#gammapy.spectrum.models.PowerLaw).

# In[9]:


pwl = PowerLaw(
    index=2. * u.Unit(''),
    amplitude=1e-12 * u.Unit('cm-2 s-1 TeV-1'),
    reference=1. * u.TeV
)


# After creating the model we run the fit by passing the `'flux_points'` and `'pwl'` objects:

# In[10]:


result_pwl = fitter.run(flux_points, pwl)


# And print the result:

# In[11]:


print(result_pwl['best-fit-model'])


# As a quick check we print the value of the fit statistics per degrees of freedom as well:

# In[12]:


print(result_pwl['statval/dof'])


# Finally we plot the data points and the best fit model:

# In[13]:


ax = flux_points.plot(energy_power=2)
result_pwl['best-fit-model'].plot(energy_range=[1e-4, 1e2] * u.TeV, ax=ax, energy_power=2)
result_pwl['best-fit-model'].plot_error(energy_range=[1e-4, 1e2] * u.TeV, ax=ax, energy_power=2)
ax.set_ylim(1e-13, 1e-11)


# ## Exponential Cut-Off Powerlaw Fit
# 
# Next we fit an [exponential cut-off power](http://docs.gammapy.org/0.7/api/gammapy.spectrum.models.ExponentialCutoffPowerLaw.html#gammapy.spectrum.models.ExponentialCutoffPowerLaw) law to the data.

# In[14]:


ecpl = ExponentialCutoffPowerLaw(
    index=2. * u.Unit(''),
    amplitude=1e-12 * u.Unit('cm-2 s-1 TeV-1'),
    reference=1. * u.TeV,
    lambda_=0. / u.TeV
)


# We run the fitter again by passing the flux points and the `ecpl` model instance:

# In[15]:


result_ecpl = fitter.run(flux_points, ecpl)
print(result_ecpl['best-fit-model'])


# In[16]:


print(result_ecpl['statval/dof'])


# We plot the data and best fit model:

# In[17]:


ax = flux_points.plot(energy_power=2)
result_ecpl['best-fit-model'].plot(energy_range=[1e-4, 1e2] * u.TeV, ax=ax, energy_power=2)
result_ecpl['best-fit-model'].plot_error(energy_range=[1e-4, 1e2] * u.TeV, ax=ax, energy_power=2)
ax.set_ylim(1e-13, 1e-11)


# ## Log-Parabola Fit
# 
# Finally we try to fit a [log-parabola](http://docs.gammapy.org/0.7/api/gammapy.spectrum.models.LogParabola.html#gammapy.spectrum.models.LogParabola) model:

# In[18]:


log_parabola = LogParabola(
    alpha=2. * u.Unit(''),
    amplitude=1e-12 * u.Unit('cm-2 s-1 TeV-1'),
    reference=1. * u.TeV,
    beta=0. * u.Unit('')
)


# In[19]:


result_log_parabola = fitter.run(flux_points, log_parabola)
print(result_log_parabola['best-fit-model'])


# In[20]:


print(result_log_parabola['statval/dof'])


# In[21]:


ax = flux_points.plot(energy_power=2)
result_log_parabola['best-fit-model'].plot(energy_range=[1e-4, 1e2] * u.TeV, ax=ax, energy_power=2)
result_log_parabola['best-fit-model'].plot_error(energy_range=[1e-4, 1e2] * u.TeV, ax=ax, energy_power=2)
ax.set_ylim(1e-13, 1e-11)


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
# * To see what's available in Gammapy, browse the [Gammapy docs](http://docs.gammapy.org/) or use the full-text search.
# * If you have any questions, ask on the mailing list .
