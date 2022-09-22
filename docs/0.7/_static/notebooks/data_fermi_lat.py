
# coding: utf-8

# # Fermi-LAT data with Gammapy
# 
# ## Introduction
# 
# This tutorial will show you how to work with pepared Fermi-LAT datasets.
# 
# The main class to load and handle the data is:
# 
# * [gammapy.dataset.FermiLATDataset](http://docs.gammapy.org/0.7/api/gammapy.datasets.FermiLATDataset.html)
# 
# 
# Additionally we will use objects of these types:
# 
# * [gammapy.data.EventList](http://docs.gammapy.org/0.7/api/gammapy.data.EventList.html) for event lists.
# * [gammapy.irf.EnergyDependentTablePSF](http://docs.gammapy.org/0.7/api/gammapy.irf.EnergyDependentTablePSF.html) for the point spread function.
# * [gammapy.cube.SkyCube](http://docs.gammapy.org/0.7/api/gammapy.cube.SkyCube.html) for the galactic diffuse background model. 
# * [gammapy.cube.SkyCubeHPX](http://docs.gammapy.org/0.7/api/gammapy.cube.SkyCubeHPX.html) for the exposure. 
# 
# * [gammapy.spectrum.models.TableModel](http://docs.gammapy.org/0.7/api/gammapy.spectrum.models.TableModel.html#gammapy.spectrum.models.TableModel) for the isotropic diffuse model.
# * [gammapy.image.FermiLATBasicImageEstimator](http://docs.gammapy.org/0.7/api/gammapy.image.FermiLATBasicImageEstimator.html) for generating a full WCS dataset, that can be used as an input for image based analyses.
# 
# 
# ## Setup
# 
# **IMPORTANT**: For this notebook you have to get the prepared datasets provided in the [gammapy-fermi-lat-data](https://github.com/gammapy/gammapy-fermi-lat-data) repository. Please follow the instructions [here](https://github.com/gammapy/gammapy-fermi-lat-data#get-the-data) to download the data and setup your environment.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from astropy import units as u
from astropy.visualization import simple_norm
from gammapy.datasets import FermiLATDataset
from gammapy.image import SkyImage, FermiLATBasicImageEstimator


# ## FermiLATDataset class
# 
# To access the prepared Fermi-LAT datasets Gammapy provides a convenience class called [FermiLATDataset](http://docs.gammapy.org/0.7/api/gammapy.datasets.FermiLATDataset.html#gammapy.datasets.FermiLATDataset). It is initialized with a path to an index configuration file, which tells the dataset class where to find the data. Once 
# the object is initialized the data can be accessed as properties of this object, which return the corresponding Gammapy data objects for event lists, sky images and point spread functions (PSF). 
# 
# So let's start with exploring the Fermi-LAT 2FHL dataset:

# In[3]:


# initialize dataset
dataset = FermiLATDataset('$GAMMAPY_FERMI_LAT_DATA/2fhl/fermi_2fhl_data_config.yaml')
print(dataset)


# ## Events
# 
# The first data member we will inspect in more detail is the event list. It can be accessed by the `dataset.events` property and returns an instance of the Gammapy [gammapy.data.EventList](http://docs.gammapy.org/0.7/api/gammapy.data.EventList.html) class:

# In[4]:


# access events data member
events = dataset.events
print(events)


# The full event data is available via the `EventList.table` property, which returns an [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table) instance. In case of the Fermi-LAT event list this contains all the additional information on positon, zenith angle, earth azimuth angle, event class, event type etc. Execute the following cell to take a look at the event list table: 

# In[5]:


events.table


# As a short analysis example we will count the number of events above a certain minimum energy: 

# In[6]:


# define energy thresholds
energies = [50, 100, 200, 400, 800, 1600] * u.GeV

n_events_above_energy = []

for energy in energies:
    n = (events.energy > energy).sum()
    n_events_above_energy.append(n)   
    print("Number of events above {0:4.0f}: {1:5.0f}".format(energy, n))


# And plot it against energy:

# In[7]:


plt.figure(figsize=(8, 5))   
events_plot = plt.plot(energies.value, n_events_above_energy, label='Fermi 2FHL events')
plt.scatter(energies.value, n_events_above_energy, s=60, c=events_plot[0].get_color())
plt.loglog()
plt.xlabel("Min. energy (GeV)")
plt.ylabel("Counts above min. energy")
plt.xlim(4E1, 3E3)
plt.ylim(1E2, 1E5)
plt.legend()


# ## PSF
# Next we will tke a closer look at the PSF. The dataset contains a precomputed PSF model for one position of the sky (in this case the Galactic center). It can be accessed by the `dataset.psf` property and returns an instance of the Gammapy [gammapy.irf.EnergyDependentTablePSF](http://docs.gammapy.org/0.7/api/gammapy.irf.EnergyDependentTablePSF.html) class:

# In[8]:


psf = dataset.psf
print(psf)


# To get an idea of the size of the PSF we check how the containment radii of the Fermi-LAT PSF vari with energy and different containment fractions:

# In[9]:


plt.figure(figsize=(8, 5))
psf.plot_containment_vs_energy(linewidth=2, fractions=[0.68, 0.95, 0.99])
plt.xlim(50, 2000)
plt.show()


# In addition we can check how the actual shape of the PSF varies with energy and compare it against the mean PSF between 50 GeV and 2000 GeV:

# In[10]:


plt.figure(figsize=(8, 5))

for energy in energies:
    psf_at_energy = psf.table_psf_at_energy(energy)
    psf_at_energy.plot_psf_vs_theta(label='PSF @ {:.0f}'.format(energy), lw=2)

erange = [50, 2000] * u.GeV
psf_mean = psf.table_psf_in_energy_band(energy_band=erange, spectral_index=2.3)
psf_mean.plot_psf_vs_theta(label='PSF Mean', lw=4, c="k", ls='--')
    
plt.xlim(1E-3, 1)
plt.ylim(1E2, 1E7)
plt.legend()


# ## Exposure
# The Fermi-LAT datatset contains the energy-dependent exposure for the whole sky stored using a HEALPIX pixelisation of the sphere. It can be accessed by the `dataset.exposure` property and returns an instance of the Gammapy `gammapy.cube.SkyCubeHPX` class:

# In[11]:


exposure = dataset.exposure
print(exposure)


# In[12]:


# define reference image using a cartesian projection
image_ref = SkyImage.empty(nxpix=360, nypix=180, binsz=1, proj='CAR')

# reproject HEALPIC sky cube
exposure_reprojected = exposure.reproject(image_ref)


# In[13]:


exposure_reprojected.show()


# You can use the slider to slide through the different energy bands.

# ## Galactic diffuse background

# The Fermi-LAT collaboration provides a galactic diffuse emission model, that can be used as a background model for
# Fermi-LAT data analysis. Currently Gammapy only supports the latest model (`gll_iem_v06.fits`). It can be accessed by the `dataset.galactic_diffuse` property and returns an instance of the Gammapy [gammapy.cube.SkyCube](http://docs.gammapy.org/0.7/api/gammapy.cube.SkyCube.html) class:

# In[14]:


galactic_diffuse = dataset.galactic_diffuse
print(galactic_diffuse)


# In[15]:


norm = simple_norm(image_ref.data, stretch='log', clip=True)
galactic_diffuse.show(norm=norm)


# Again you can use the slider to slide through the different energy bands. E.g. note how the Fermi-Bubbles become more present at higher energies (higher value of idx).

# ## Isotropic diffuse background
# 
# Additionally to the galactic diffuse model, there is an isotropic diffuse component. It can be accessed by the `dataset.isotropic_diffuse` property and returns an instance of the Gammapy [gammapy.spectrum.models.TableModel](http://docs.gammapy.org/0.7/api/gammapy.spectrum.models.TableModel.html#gammapy.spectrum.models.TableModel) class:

# In[16]:


isotropic_diffuse = dataset.isotropic_diffuse
print(isotropic_diffuse)


# We can plot the model in the energy range between 50 GeV and 2000 GeV:

# In[17]:


erange = [50, 2000] * u.GeV
isotropic_diffuse.plot(erange)


# ## Estimating basic input sky images for high level analyses
# Finally we'd like to use the prepared 2FHL dataset to generate a set of basic sky images, that a can be used as input for high level analyses, e.g. morphology fits, region based flux measurements, computation of significance images etc.
# For this purpose Gammapy provides a convenience class called [gammapy.image.FermiLATBasicImageEstimator](http://docs.gammapy.org/0.7/api/gammapy.image.FermiLATBasicImageEstimator.html). First we define a reference image, that specifies the region we'd like to analyse. In this case we choose the Vela region.

# In[18]:


image_ref = SkyImage.empty(
    nxpix=360, nypix=180,
    binsz=0.05,
    xref=265, yref=0,
)


# Next we choose the energy range and initialize the `FermiLATBasicImageEstimator` object:

# In[19]:


emin, emax = [50, 2000] * u.GeV
image_estimator = FermiLATBasicImageEstimator(
    reference=image_ref,
    emin=emin, emax=emax,
)


# Finally we run the image estimation by calling `.run()` and parsing the dataset object:

# In[20]:


images_basic = image_estimator.run(dataset)


# The image estimator now computes a set of sky images for the reference region and energy range we defined above. The result `images_basic` is a [gammapy.image.SkyImageList](http://docs.gammapy.org/0.7/api/gammapy.image.SkyImageList.html) object containing the following images:
# 
# * **counts**: counts image containing the binned event list
# * **background**: predicted number of background counts computed from the sum of the galactic and isotropic diffuse model, given the exposure.
# * **exposure**: integrated exposure assuming a powerlaw with spectral index 2.3 in the given energy range
# * **excess**: backround substracted counts image
# * **flux**: measured flux, computed from excess divided by exposure
# * **psf**: sky image of the exposure weighted mean PSF in the given energy range
#     
# You can check the contained images as following:

# In[21]:


images_basic.names


# To check whether the image estimation was succesfull we'll take a look at the flux image, smoothing it in advance with a Gaussian kernel of 0.2 deg:

# In[22]:


smoothed_flux = images_basic['flux'].smooth(
    kernel='gauss', radius=0.2 * u.deg)
smoothed_flux.show()


# ## Exercises
# 
# - Try to reproject the exposure using an `AIT` projection.
# - Try to find the spectral index of the isotropic diffuse model using a method off the `TableModel` instance.
# - Compute basic sky images for different regions (e.g. Galactic Center) and energy ranges
# 

# ## What next?
# 
# In this tutorial we have learned how to access and check Fermi-LAT data.
# 
# Next you could do:
# * image analysis
# * spectral analysis
# * cube analysis
# * time analysis
# * source detection
