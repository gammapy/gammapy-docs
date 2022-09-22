
# coding: utf-8

# # Getting started with Gammapy
# 
# ## Introduction
# 
# This is a getting started tutorial for [Gammapy](http://docs.gammapy.org/).
# 
# In this tutorial we will use the [Second Fermi-LAT Catalog of High-Energy Sources (2FHL) catalog](http://fermi.gsfc.nasa.gov/ssc/data/access/lat/2FHL/), corresponding event list and images to learn how to work with some of the central Gammapy data structures.
# 
# We will cover the following topics:
# 
# * **Sky images**
#   * We will learn how to handle image based data with gammapy using a Fermi-LAT 2FHL example image. We will work with the following classes:
#     - [gammapy.image.SkyImage](http://docs.gammapy.org/0.7/api/gammapy.image.SkyImage.html)
#     - [astropy.coordinates.SkyCoord](http://astropy.readthedocs.io/en/latest/coordinates/index.html)
#     - [numpy.ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html)
# 
# * **Event lists**
#   * We will learn how to handle event lists with Gammapy. Important for this are the following classes: 
#     - [gammapy.data.EventList](http://docs.gammapy.org/0.7/api/gammapy.data.EventList.html)
#     - [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table)
# 
# * **Source catalogs**
#   * We will show how to load source catalogs with Gammapy and explore the data using the following classes:
#     - [gammapy.catalog.SourceCatalog](http://docs.gammapy.org/0.7/api/gammapy.catalog.SourceCatalog.html), specifically [gammapy.catalog.SourceCatalog2FHL](http://docs.gammapy.org/0.7/api/gammapy.catalog.SourceCatalog2FHL.html)
#     - [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table)
# 
# * **Spectral models and flux points**
#   * We will pick an example source and show how to plot its spectral model and flux points. For this we will use the following classes:
#     - [gammapy.spectrum.SpectralModel](http://docs.gammapy.org/0.7/api/gammapy.spectrum.models.SpectralModel.html), specifically the [PowerLaw2](http://docs.gammapy.org/0.7/api/gammapy.spectrum.models.PowerLaw2.html) model.
#     - [gammapy.spectrum.FluxPoints](http://docs.gammapy.org/0.7/api/gammapy.spectrum.FluxPoints.html#gammapy.spectrum.FluxPoints)
#     - [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table)
# 
# If you're not yet familiar with the listed Astropy classes, maybe check out the [Astropy introduction for Gammapy users](astropy_introduction.ipynb) first.

# ## Setup
# 
# **Important**: to run this tutorial the environment variable `GAMMAPY_EXTRA` must be defined and point to the directory, where the gammapy-extra is respository located on your machine. To check whether your setup is correct you can execute the following cell:
# 
# 

# In[1]:


import os

path = os.path.expandvars('$GAMMAPY_EXTRA/datasets')

if not os.path.exists(path):
    raise Exception('gammapy-extra repository not found!')
else:
    print('Great your setup is correct!')


# In case you encounter an error, you can un-comment and execute the following cell to continue. But we recommend to set up your enviroment correctly as decribed [here](http://docs.gammapy.org/0.7/datasets/index.html#gammapy-extra) after you are done with this notebook.

# In[2]:


#os.environ['GAMMAPY_EXTRA'] = os.path.join(os.getcwd(), '..')


# Now we can continue with the usual IPython notebooks and Python imports:

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[4]:


import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm


# ## Sky images
# 
# The central data structure to work with image based data in Gammapy is the [SkyImage](http://docs.gammapy.org/0.7/api/gammapy.image.SkyImage.html) class. It combines the raw data with world coordinate (WCS) information, FITS I/O functionality and convenience methods, that allow easy handling, processing and plotting of image based data. 
# 
# In this section we will learn how to:
# 
# * Read sky images from FITS files
# * Smooth images
# * Plot images
# * Cutout parts from images
# * Reproject images to different WCS
# 
# The `SkyImage` class is part of the [gammapy.image](http://docs.gammapy.org/0.7/image/index.html) submodule. So we will start by importing it from there:

# In[5]:


from gammapy.image import SkyImage


# As a first example, we will read a FITS file from a prepared Fermi-LAT 2FHL dataset:

# In[6]:


vela_2fhl = SkyImage.read("$GAMMAPY_EXTRA/datasets/fermi_2fhl/fermi_2fhl_vela.fits.gz", hdu='COUNTS')


# As the FITS file `fermi_2fhl_vela.fits.gz` contains mutiple image extensions and a `SkyImage` can only represent a single image, we explicitely specify to read the extension called 'Counts'. Let's print the image to get some basic information about it:

# In[7]:


print(vela_2fhl)


# The shape of the image is 320x180 pixel, the data unit is counts ('ct') and it is defined using a cartesian projection in galactic coordinates.
# 
# Let's take a closer look a the `.data` attribute:

# In[8]:


vela_2fhl.data


# That looks familiar! It just an *ordinary* 2 dimensional numpy array,  which means you can apply any known numpy method to it:

# In[9]:


print('Total number of counts in the image: {:.0f}'.format(vela_2fhl.data.sum()))


# To show the image on the screen we can use the [SkyImage.show()](http://docs.gammapy.org/0.7/api/gammapy.image.SkyImage.html#gammapy.image.SkyImage.show) method. It basically calls [plt.imshow](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow), passing the `vela_2fhl.data` attribute but in addition handles axis with world coordinates using [wcsaxes](https://wcsaxes.readthedocs.io/en/latest/) and defines some defaults for nicer plots (e.g. the colormap 'afmhot'):

# In[10]:


vela_2fhl.show()


# To make the structures in the image more visible we will smooth the data using a Gausian kernel with a radius of 0.5 deg. Again `SkyImage.smooth()` is a wrapper around existing functionality from the scientific Python libraries. In this case it is Scipy's [gaussian_filter](https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.gaussian_filter.html) method. For convenience the kernel shape can be specified with as string and the smoothing radius with a quantity. It returns again a `SkyImage` object, that we can plot directly the same way we did above:

# In[11]:


# smooth counts image with gaussian kernel of 0.5 deg
vela_2fhl_smoothed = vela_2fhl.smooth(kernel='gauss', radius=0.5 * u.deg)
vela_2fhl_smoothed.show()


# The smoothed plot already looks much nicer, but still the image is rather large. As we are mostly interested in the inner part of the image, we will cut out a quadratic region of the size 9 deg x 9 deg around Vela. Therefore we use [SkyImage.cutout()](http://docs.gammapy.org/0.7/api/gammapy.image.SkyImage.html#gammapy.image.SkyImage.cutout), which wraps Astropy's [Cutout2D](http://docs.astropy.org/en/stable/api/astropy.nddata.Cutout2D.html). Again it returns a `SkyImage` object:

# In[12]:


# define center and size of the cutout region
center = SkyCoord(265.0, -2.0, unit='deg', frame='galactic')
size = 9.0 * u.deg

vela_2fhl_cutout = vela_2fhl_smoothed.cutout(center, size)
vela_2fhl_cutout.show()


# To make this exercise a bit more scientifically useful, we will load a second image containing WMAP data from the same region:

# In[13]:


vela_wmap = SkyImage.read("$GAMMAPY_EXTRA/datasets/images/Vela_region_WMAP_K.fits")

# define a norm to stretch the data, so it is better visible
norm = simple_norm(vela_wmap.data, stretch='sqrt', max_percent=99.9)
vela_wmap.show(cmap='viridis', norm=norm)


# In order to make the structures in the data better visible we used the [simple_norm()](http://docs.astropy.org/en/stable/api/astropy.visualization.mpl_normalize.simple_norm.html#astropy.visualization.mpl_normalize.simple_norm) method, which allows to stretch the data for plotting. This is very similar to the methods that e.g. `ds9` provides. In addition we used a different colormap called 'viridis'. An overview of different colomaps can be found [here](http://matplotlib.org/examples/color/colormaps_reference.html). 
# 
# Now let's check the details of the WMAP image:

# In[14]:


print(vela_wmap)


# As you can see it is defined using a tangential projection and ICRS coordinates, which is different from the projection used for the `vela_2fhl` image. To compare both images we have to reproject one image to the WCS of the other. This can be achieved with the [SkyImage.reproject()](http://docs.gammapy.org/0.7/api/gammapy.image.SkyImage.html#gammapy.image.SkyImage.reproject) method: 

# In[15]:


# reproject WMAP image
vela_wmap_reprojected = vela_wmap.reproject(vela_2fhl)

# cutout part we're interested in
vela_wmap_reprojected_cutout = vela_wmap_reprojected.cutout(center, size)
vela_wmap_reprojected_cutout.show(cmap='viridis', norm=norm)


# Finally we will combine both images in single plot, by plotting WMAP contours on top of the smoothed Fermi-LAT 2FHL image:
# 

# In[16]:


fig, ax, _ = vela_2fhl_cutout.plot()
ax.contour(vela_wmap_reprojected_cutout.data, cmap='Blues')


# ### Exercises
# 
# * Add a marker and circle at the Vela pulsar position (you can find examples in the WCSAxes [documentation](https://wcsaxes.readthedocs.io/en/latest/overlays.html)).
# * Find the maximum brightness location in the WMAP image. The methods [np.argmax()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html) and [SkyImage.wcs_pixel_to_skycoord()](http://docs.gammapy.org/0.7/api/gammapy.image.SkyImage.html#gammapy.image.SkyImage.wcs_pixel_to_skycoord) might be helpful. Try to identify the source.
# 
# 
# 

# ## Event lists
# 
# Almost any high-level gamma-ray data analysis starts with the raw measured counts data, which is stored in event lists. In Gammapy event lists are represented by the [gammapy.data.EventList](http://docs.gammapy.org/0.7/api/gammapy.data.EventList.html) class. 
# 
# In this section we will learn how to:
# 
# * Read event lists from FITS files
# * Access and work with the `EventList` attributes such as `.table` and `.energy` 
# * Filter events lists using convenience methods
# 
# Let's start with the import from the [gammapy.data](http://docs.gammapy.org/0.7/data/index.html) submodule:

# In[17]:


from gammapy.data import EventList


# Very similar to the `SkyImage` class an event list can be created, by passing a filename to the `.read()` method:

# In[18]:


events_2fhl = EventList.read('$GAMMAPY_EXTRA/datasets/fermi_2fhl/2fhl_events.fits.gz')


# This time the actual data is stored as an [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table) object. It can be accessed with `.table` attribute: 

# In[19]:


events_2fhl.table


# Let's try to find the total number of events contained int the list. This doesn't work:
# 

# In[20]:


print('Total number of events: {}'.format(len(events_2fhl.table)))


# Because Gammapy objects don't redefine properties, that are accessible from the underlying Astropy of Numpy data object.
# 
# So in this case of course we can directly use the `.table` attribute to find the total number of events:

# In[21]:


print('Total number of events: {}'.format(len(events_2fhl.table)))


# And we can access any other attribute of the `Table` object as well:

# In[22]:


events_2fhl.table.colnames


# For convenience we can access the most important event parameters as properties on the `EventList` objects. The attributes will return corresponding Astropy objects to represent the data, such as [astropy.units.Quantity](http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html#astropy.units.Quantity), [astropy.coordinates.SkyCoord](http://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) or [astropy.time.Time](http://docs.astropy.org/en/stable/api/astropy.time.Time.html#astropy.time.Time) objects:

# In[23]:


events_2fhl.energy.to('GeV')


# In[24]:


events_2fhl.galactic
#events_2fhl.radec


# In[25]:


events_2fhl.time


# In addition `EventList` provides convenience methods to filter the event lists. One possible use case is to find the highest energy event within a radius of 0.5 deg around the vela position:

# In[26]:


# select all events within a radius of 0.5 deg around center
events_vela_2fhl = events_2fhl.select_sky_cone(center=center, radius=0.5 * u.deg)

# sort events by energy
events_vela_2fhl.table.sort('ENERGY')

# and show highest energy photon
events_vela_2fhl.energy[-1].to('GeV')


# ### Exercises
# 
# * Make a counts energy spectrum for the galactic center region, within a radius of 10 deg.

# ## Source catalogs
# 
# Gammapy provides a convenient interface to access and work with catalog based data. 
# 
# In this section we will learn how to:
# 
# * Load builtins catalogs from [gammapy.catalog](http://docs.gammapy.org/0.7/catalog/index.html)
# * Sort and index the underlying Astropy tables
# * Access data from individual sources
# 
# Let's start with importing the 2FHL catalog object from the [gammapy.catalog](http://docs.gammapy.org/0.7/catalog/index.html) submodule:

# In[27]:


from gammapy.catalog import SourceCatalog2FHL


# First we initialize the Fermi-LAT 2FHL catalog and directly take a look at the `.table` attribute:

# In[28]:


fermi_2fhl = SourceCatalog2FHL('$GAMMAPY_EXTRA/datasets/catalogs/fermi/gll_psch_v08.fit.gz')
fermi_2fhl.table


# This looks very familiar again. The data is just stored as an [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table) object. We have all the methods and attributes of the `Table` object available. E.g. we can sort the underlying table by `TS` to find the top 5 most significant sources:
# 
# 

# In[29]:


# sort table by TS
fermi_2fhl.table.sort('TS')

# invert the order to find the highest values and take the top 5 
top_five_TS_2fhl = fermi_2fhl.table[::-1][:5]

# print the top five significant sources with association and source class
top_five_TS_2fhl[['Source_Name', 'ASSOC', 'CLASS']]


# If you are interested in the data of an individual source you can access the information from catalog using the name of the source or any alias source name that is defined in the catalog:

# In[30]:


mkn_421_2fhl = fermi_2fhl['2FHL J1104.4+3812']

# or use any alias source name that is defined in the catalog
mkn_421_2fhl = fermi_2fhl['Mkn 421']
print(mkn_421_2fhl.data['TS'])


# ### Exercises
# 
# * Try to load the Fermi-LAT 3FHL catalog and check the total number of sources it contains.
# * Select all the sources from the 2FHL catalog which are contained in the Vela region. Add markers for all these sources and try to add labels with the source names. The methods [SkyImage.contains()](http://docs.gammapy.org/0.7/api/gammapy.image.SkyImage.html#gammapy.image.SkyImage.contains) and [ax.text()](http://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.text.html#matplotlib.axes.Axes.text) might be helpful.
# * Try to find the source class of the object at position ra=68.6803, dec=9.3331
#  

# ## Spectral models and flux points
# 
# In the previous section we learned how access basic data from individual sources in the catalog. Now we will go one step further and explore the full spectral information of sources. We will learn how to:
# 
# * Plot spectral models
# * Compute integral and energy fluxes
# * Read and plot flux points
# 
# As a first example we will start with the Crab Nebula:

# In[31]:


crab_2fhl = fermi_2fhl['Crab']
print(crab_2fhl.spectral_model)


# The `crab_2fhl.spectral_model` is an instance of the [gammapy.spectrum.models.PowerLaw2](http://docs.gammapy.org/0.7/api/gammapy.spectrum.models.PowerLaw2.html#gammapy.spectrum.models.PowerLaw2) model, with the parameter values and errors taken from the 2FHL catalog. 
# 
# Let's plot the spectral model in the energy range between 50 GeV and 2000 GeV:

# In[32]:


ax_crab_2fhl = crab_2fhl.spectral_model.plot(
    energy_range=[50, 2000] * u.GeV, energy_power=0)


# We assign the return axes object to variable called `ax_crab_2fhl`, because we will re-use it later to plot the flux points on top.
# 
# To compute the differential flux at 100 GeV we can simply call the model like normal Python function and convert to the desired units:

# In[33]:


crab_2fhl.spectral_model(100 * u.GeV).to('cm-2 s-1 GeV-1')


# Next we can compute the integral flux of the Crab between 50 GeV and 2000 GeV:

# In[34]:


crab_2fhl.spectral_model.integral(
    emin=50 * u.GeV, emax=2000 * u.GeV,
).to('cm-2 s-1')


# We can easily convince ourself, that it corresponds to the value given in the Fermi-LAT 2FHL catalog:

# In[35]:


crab_2fhl.data['Flux50']


# In addition we can compute the energy flux between 50 GeV and 2000 GeV:

# In[36]:


crab_2fhl.spectral_model.energy_flux(
    emin=50 * u.GeV, emax=2000 * u.GeV,
).to('erg cm-2 s-1')


# Next we will access the flux points data of the Crab:

# In[37]:


print(crab_2fhl.flux_points)


# If you want to learn more about the different flux point formats you can read the specification [here](http://gamma-astro-data-formats.readthedocs.io/en/latest/results/flux_points/index.html#flux-points).
# 
# No we can check again the underlying astropy data structure by accessing the `.table` attribute:

# In[38]:


crab_2fhl.flux_points.table


# Finally let's combine spectral model and flux points in a single plot and scale with `energy_power=2` to obtain the spectral energy distribution:

# In[39]:


ax = crab_2fhl.spectral_model.plot(
    energy_range=[50, 2000] * u.GeV, energy_power=2,
)
crab_2fhl.flux_points.plot(ax=ax, energy_power=2)


# ### Exercises
# 
# * Plot the spectral model and flux points for PKS 2155-304 for the 3FGL and 2FHL catalogs. Try to plot the error of the model (aka "Butterfly") as well. Note this requires the [uncertainties package](https://pythonhosted.org/uncertainties/) to be installed on your machine.
# 

# ## What next?
# 
# This was a quick introduction to some of the high-level classes in Astropy and Gammapy.
# 
# * To learn more about those classes, go to the API docs (links are in the introduction at the top).
# * To learn more about other parts of Gammapy (e.g. Fermi-LAT and TeV data analysis), check out the other tutorial notebooks.
# * To see what's available in Gammapy, browse the [Gammapy docs](http://docs.gammapy.org/) or use the full-text search.
# * If you have any questions, ask on the mailing list.
