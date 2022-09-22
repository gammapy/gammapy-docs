
# coding: utf-8

# # Getting started with Gammapy
# 
# ## Introduction
# 
# This is a getting started tutorial for [Gammapy](https://docs.gammapy.org/).
# 
# In this tutorial we will use the [Second Fermi-LAT Catalog of High-Energy Sources (3FHL) catalog](http://fermi.gsfc.nasa.gov/ssc/data/access/lat/3FHL/), corresponding event list and images to learn how to work with some of the central Gammapy data structures.
# 
# We will cover the following topics:
# 
# * **Sky maps**
#   * We will learn how to handle image based data with gammapy using a Fermi-LAT 3FHL example image. We will work with the following classes:
#     - [gammapy.maps.WcsNDMap](https://docs.gammapy.org/0.11/api/gammapy.maps.WcsNDMap.html)
#     - [astropy.coordinates.SkyCoord](http://astropy.readthedocs.io/en/latest/coordinates/index.html)
#     - [numpy.ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html)
# 
# * **Event lists**
#   * We will learn how to handle event lists with Gammapy. Important for this are the following classes: 
#     - [gammapy.data.EventList](https://docs.gammapy.org/0.11/api/gammapy.data.EventList.html)
#     - [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table)
# 
# * **Source catalogs**
#   * We will show how to load source catalogs with Gammapy and explore the data using the following classes:
#     - [gammapy.catalog.SourceCatalog](https://docs.gammapy.org/0.11/api/gammapy.catalog.SourceCatalog.html), specifically [gammapy.catalog.SourceCatalog3FHL](https://docs.gammapy.org/0.11/api/gammapy.catalog.SourceCatalog3FHL.html)
#     - [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table)
# 
# * **Spectral models and flux points**
#   * We will pick an example source and show how to plot its spectral model and flux points. For this we will use the following classes:
#     - [gammapy.spectrum.SpectralModel](https://docs.gammapy.org/0.11/api/gammapy.spectrum.models.SpectralModel.html), specifically the [PowerLaw2](https://docs.gammapy.org/0.11/api/gammapy.spectrum.models.PowerLaw2.html) model.
#     - [gammapy.spectrum.FluxPoints](https://docs.gammapy.org/0.11/api/gammapy.spectrum.FluxPoints.html#gammapy.spectrum.FluxPoints)
#     - [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table)
# 
# If you're not yet familiar with the listed Astropy classes, maybe check out the [Astropy introduction for Gammapy users](astropy_introduction.ipynb) first.

# ## Setup
# 
# **Important**: to run this tutorial the environment variable `GAMMAPY_DATA` must be defined and point to the directory on your machine where the datasets needed are placed. To check whether your setup is correct you can execute the following cell:
# 
# 

# In[ ]:


import os

path = os.path.expandvars("$GAMMAPY_DATA")

if not os.path.exists(path):
    raise Exception("gammapy-data repository not found!")
else:
    print("Great your setup is correct!")


# In case you encounter an error, you can un-comment and execute the following cell to continue. But we recommend to set up your enviroment correctly as decribed [here](https://docs.gammapy.org/0.11/getting-started.html#download-tutorials) after you are done with this notebook.

# In[ ]:


# os.environ['GAMMAPY_DATA'] = os.path.join(os.getcwd(), '..')


# Now we can continue with the usual IPython notebooks and Python imports:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm


# ## Maps
# 
# The [gammapy.maps](https://docs.gammapy.org/0.11/maps) package contains classes to work with sky images and cubes.
# 
# In this section, we will use a simple 2D sky image and will learn how to:
# 
# * Read sky images from FITS files
# * Smooth images
# * Plot images
# * Cutout parts from images
# * Reproject images to different WCS

# In[ ]:


from gammapy.maps import Map

gc_3fhl = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts.fits.gz")


# The image is a ``WCSNDMap`` object:

# In[ ]:


gc_3fhl


# The shape of the image is 400 x 200 pixel and it is defined using a cartesian projection in galactic coordinates.
# 
# The ``geom`` attribute is a ``WcsGeom`` object:

# In[ ]:


gc_3fhl.geom


# Let's take a closer look a the `.data` attribute:

# In[ ]:


gc_3fhl.data


# That looks familiar! It just an *ordinary* 2 dimensional numpy array,  which means you can apply any known numpy method to it:

# In[ ]:


print("Total number of counts in the image: {:.0f}".format(gc_3fhl.data.sum()))


# To show the image on the screen we can use the ``plot`` method. It basically calls [plt.imshow](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow), passing the `gc_3fhl.data` attribute but in addition handles axis with world coordinates using [wcsaxes](https://wcsaxes.readthedocs.io/en/latest/) and defines some defaults for nicer plots (e.g. the colormap 'afmhot'):

# In[ ]:


gc_3fhl.plot(stretch="sqrt");


# To make the structures in the image more visible we will smooth the data using a Gausian kernel with a radius of 0.5 deg. Again `smooth()` is a wrapper around existing functionality from the scientific Python libraries. In this case it is Scipy's [gaussian_filter](https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.gaussian_filter.html) method. For convenience the kernel shape can be specified with as string and the smoothing radius with a quantity. It returns again a map object, that we can plot directly the same way we did above:

# In[ ]:


gc_3fhl_smoothed = gc_3fhl.smooth(kernel="gauss", width=0.2 * u.deg)


# In[ ]:


gc_3fhl_smoothed.plot(stretch="sqrt");


# The smoothed plot already looks much nicer, but still the image is rather large. As we are mostly interested in the inner part of the image, we will cut out a quadratic region of the size 9 deg x 9 deg around Vela. Therefore we use ``Map.cutout`` to make a cutout map:

# In[ ]:


# define center and size of the cutout region
center = SkyCoord(0, 0, unit="deg", frame="galactic")
gc_3fhl_cutout = gc_3fhl_smoothed.cutout(center, 9 * u.deg)
gc_3fhl_cutout.plot(stretch="sqrt");


# For a more detailed introdcution to `ganmmapy.maps`, take a look a the [intro_maps.ipynb](intro_maps.ipynb) notebook.
# 
# ### Exercises
# 
# * Add a marker and circle at the position of `Sag A*` (you can find examples in the WCSAxes [documentation](https://wcsaxes.readthedocs.io/en/latest/overlays.html)).

# ## Event lists
# 
# Almost any high-level gamma-ray data analysis starts with the raw measured counts data, which is stored in event lists. In Gammapy event lists are represented by the [gammapy.data.EventList](https://docs.gammapy.org/0.11/api/gammapy.data.EventList.html) class. 
# 
# In this section we will learn how to:
# 
# * Read event lists from FITS files
# * Access and work with the `EventList` attributes such as `.table` and `.energy` 
# * Filter events lists using convenience methods
# 
# Let's start with the import from the [gammapy.data](https://docs.gammapy.org/0.11/data/index.html) submodule:

# In[ ]:


from gammapy.data import EventList


# Very similar to the sky map class an event list can be created, by passing a filename to the `.read()` method:

# In[ ]:


events_3fhl = EventList.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz"
)


# This time the actual data is stored as an [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table) object. It can be accessed with `.table` attribute: 

# In[ ]:


events_3fhl.table


# You can do *len* over event_3fhl.table to find the total number of events.

# In[ ]:


print("Total number of events: {}".format(len(events_3fhl.table)))


# And we can access any other attribute of the `Table` object as well:

# In[ ]:


events_3fhl.table.colnames


# For convenience we can access the most important event parameters as properties on the `EventList` objects. The attributes will return corresponding Astropy objects to represent the data, such as [astropy.units.Quantity](http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html#astropy.units.Quantity), [astropy.coordinates.SkyCoord](http://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) or [astropy.time.Time](http://docs.astropy.org/en/stable/api/astropy.time.Time.html#astropy.time.Time) objects:

# In[ ]:


events_3fhl.energy.to("GeV")


# In[ ]:


events_3fhl.galactic
# events_3fhl.radec


# In[ ]:


events_3fhl.time


# In addition `EventList` provides convenience methods to filter the event lists. One possible use case is to find the highest energy event within a radius of 0.5 deg around the vela position:

# In[ ]:


# select all events within a radius of 0.5 deg around center
events_gc_3fhl = events_3fhl.select_sky_cone(center=center, radius=0.5 * u.deg)

# sort events by energy
events_gc_3fhl.table.sort("ENERGY")

# and show highest energy photon
events_gc_3fhl.energy[-1].to("GeV")


# ### Exercises
# 
# * Make a counts energy spectrum for the galactic center region, within a radius of 10 deg.

# ## Source catalogs
# 
# Gammapy provides a convenient interface to access and work with catalog based data. 
# 
# In this section we will learn how to:
# 
# * Load builtins catalogs from [gammapy.catalog](https://docs.gammapy.org/0.11/catalog/index.html)
# * Sort and index the underlying Astropy tables
# * Access data from individual sources
# 
# Let's start with importing the 3FHL catalog object from the [gammapy.catalog](https://docs.gammapy.org/0.11/catalog/index.html) submodule:

# In[ ]:


from gammapy.catalog import SourceCatalog3FHL


# First we initialize the Fermi-LAT 3FHL catalog and directly take a look at the `.table` attribute:

# In[ ]:


fermi_3fhl = SourceCatalog3FHL()
fermi_3fhl.table


# This looks very familiar again. The data is just stored as an [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table) object. We have all the methods and attributes of the `Table` object available. E.g. we can sort the underlying table by `Signif_Avg` to find the top 5 most significant sources:
# 
# 

# In[ ]:


# sort table by significance
fermi_3fhl.table.sort("Signif_Avg")

# invert the order to find the highest values and take the top 5
top_five_TS_3fhl = fermi_3fhl.table[::-1][:5]

# print the top five significant sources with association and source class
top_five_TS_3fhl[["Source_Name", "ASSOC1", "ASSOC2", "CLASS", "Signif_Avg"]]


# If you are interested in the data of an individual source you can access the information from catalog using the name of the source or any alias source name that is defined in the catalog:

# In[ ]:


mkn_421_3fhl = fermi_3fhl["3FHL J1104.4+3812"]

# or use any alias source name that is defined in the catalog
mkn_421_3fhl = fermi_3fhl["Mkn 421"]
print(mkn_421_3fhl.data["Signif_Avg"])


# ### Exercises
# 
# * Try to load the Fermi-LAT 2FHL catalog and check the total number of sources it contains.
# * Select all the sources from the 2FHL catalog which are contained in the Galactic Center region. The methods [`WcsGeom.contains()`](https://docs.gammapy.org/stable/api/gammapy.maps.WcsGeom.html#gammapy.maps.WcsGeom.contains) and [`SourceCatalog.positions`](https://docs.gammapy.org/stable/api/gammapy.catalog.SourceCatalog.html#gammapy.catalog.SourceCatalog.positions) might be helpful for this. Add markers for all these sources and try to add labels with the source names. The function [ax.text()](http://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.text.html#matplotlib.axes.Axes.text) might be also helpful.
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

# In[ ]:


crab_3fhl = fermi_3fhl["Crab Nebula"]
print(crab_3fhl.spectral_model)


# The `crab_3fhl.spectral_model` is an instance of the [gammapy.spectrum.models.PowerLaw2](https://docs.gammapy.org/0.11/api/gammapy.spectrum.models.PowerLaw2.html#gammapy.spectrum.models.PowerLaw2) model, with the parameter values and errors taken from the 3FHL catalog. 
# 
# Let's plot the spectral model in the energy range between 10 GeV and 2000 GeV:

# In[ ]:


ax_crab_3fhl = crab_3fhl.spectral_model.plot(
    energy_range=[10, 2000] * u.GeV, energy_power=0
)


# We assign the return axes object to variable called `ax_crab_3fhl`, because we will re-use it later to plot the flux points on top.
# 
# To compute the differential flux at 100 GeV we can simply call the model like normal Python function and convert to the desired units:

# In[ ]:


crab_3fhl.spectral_model(100 * u.GeV).to("cm-2 s-1 GeV-1")


# Next we can compute the integral flux of the Crab between 10 GeV and 2000 GeV:

# In[ ]:


crab_3fhl.spectral_model.integral(emin=10 * u.GeV, emax=2000 * u.GeV).to(
    "cm-2 s-1"
)


# We can easily convince ourself, that it corresponds to the value given in the Fermi-LAT 3FHL catalog:

# In[ ]:


crab_3fhl.data["Flux"]


# In addition we can compute the energy flux between 10 GeV and 2000 GeV:

# In[ ]:


crab_3fhl.spectral_model.energy_flux(emin=10 * u.GeV, emax=2000 * u.GeV).to(
    "erg cm-2 s-1"
)


# Next we will access the flux points data of the Crab:

# In[ ]:


print(crab_3fhl.flux_points)


# If you want to learn more about the different flux point formats you can read the specification [here](http://gamma-astro-data-formats.readthedocs.io/en/latest/results/flux_points/index.html#flux-points).
# 
# No we can check again the underlying astropy data structure by accessing the `.table` attribute:

# In[ ]:


crab_3fhl.flux_points.table


# Finally let's combine spectral model and flux points in a single plot and scale with `energy_power=2` to obtain the spectral energy distribution:

# In[ ]:


ax = crab_3fhl.spectral_model.plot(
    energy_range=[10, 2000] * u.GeV, energy_power=2
)
crab_3fhl.flux_points.to_sed_type("dnde").plot(ax=ax, energy_power=2);


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
# * To see what's available in Gammapy, browse the [Gammapy docs](https://docs.gammapy.org/) or use the full-text search.
# * If you have any questions, ask on the mailing list.
