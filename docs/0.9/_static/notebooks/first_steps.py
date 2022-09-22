
# coding: utf-8

# # Getting started with Gammapy
# 
# ## Introduction
# 
# This is a getting started tutorial for [Gammapy](https://docs.gammapy.org/).
# 
# In this tutorial we will use the [Second Fermi-LAT Catalog of High-Energy Sources (2FHL) catalog](http://fermi.gsfc.nasa.gov/ssc/data/access/lat/2FHL/), corresponding event list and images to learn how to work with some of the central Gammapy data structures.
# 
# We will cover the following topics:
# 
# * **Sky maps**
#   * We will learn how to handle image based data with gammapy using a Fermi-LAT 2FHL example image. We will work with the following classes:
#     - [gammapy.maps.WcsNDMap](https://docs.gammapy.org/0.9/api/gammapy.maps.WcsNDMap.html)
#     - [astropy.coordinates.SkyCoord](http://astropy.readthedocs.io/en/latest/coordinates/index.html)
#     - [numpy.ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html)
# 
# * **Event lists**
#   * We will learn how to handle event lists with Gammapy. Important for this are the following classes: 
#     - [gammapy.data.EventList](https://docs.gammapy.org/0.9/api/gammapy.data.EventList.html)
#     - [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table)
# 
# * **Source catalogs**
#   * We will show how to load source catalogs with Gammapy and explore the data using the following classes:
#     - [gammapy.catalog.SourceCatalog](https://docs.gammapy.org/0.9/api/gammapy.catalog.SourceCatalog.html), specifically [gammapy.catalog.SourceCatalog2FHL](https://docs.gammapy.org/0.9/api/gammapy.catalog.SourceCatalog2FHL.html)
#     - [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table)
# 
# * **Spectral models and flux points**
#   * We will pick an example source and show how to plot its spectral model and flux points. For this we will use the following classes:
#     - [gammapy.spectrum.SpectralModel](https://docs.gammapy.org/0.9/api/gammapy.spectrum.models.SpectralModel.html), specifically the [PowerLaw2](https://docs.gammapy.org/0.9/api/gammapy.spectrum.models.PowerLaw2.html) model.
#     - [gammapy.spectrum.FluxPoints](https://docs.gammapy.org/0.9/api/gammapy.spectrum.FluxPoints.html#gammapy.spectrum.FluxPoints)
#     - [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table)
# 
# If you're not yet familiar with the listed Astropy classes, maybe check out the [Astropy introduction for Gammapy users](astropy_introduction.ipynb) first.

# ## Setup
# 
# **Important**: to run this tutorial the environment variable `GAMMAPY_EXTRA` must be defined and point to the directory, where the gammapy-extra is respository located on your machine. To check whether your setup is correct you can execute the following cell:
# 
# 

# In[ ]:


import os

path = os.path.expandvars("$GAMMAPY_DATA")

if not os.path.exists(path):
    raise Exception("gammapy-data repository not found!")
else:
    print("Great your setup is correct!")


# In case you encounter an error, you can un-comment and execute the following cell to continue. But we recommend to set up your enviroment correctly as decribed [here](https://docs.gammapy.org/0.9/datasets/index.html#gammapy-extra) after you are done with this notebook.

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
# The [gammapy.maps](https://docs.gammapy.org/0.9/maps) package contains classes to work with sky images and cubes.
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

vela_2fhl = Map.read(
    "$GAMMAPY_DATA/fermi_2fhl/fermi_2fhl_vela.fits.gz", hdu="COUNTS"
)


# As the FITS file `fermi_2fhl_vela.fits.gz` contains mutiple image extensions and a map can only represent a single image, we explicitely specify to read the extension called 'COUNTS'.
# 
# The image is a ``WCSNDMap`` object:

# In[ ]:


vela_2fhl


# The shape of the image is 320x180 pixel and it is defined using a cartesian projection in galactic coordinates.
# 
# The ``geom`` attribute is a ``WcsGeom`` object:

# In[ ]:


vela_2fhl.geom


# Let's take a closer look a the `.data` attribute:

# In[ ]:


vela_2fhl.data


# That looks familiar! It just an *ordinary* 2 dimensional numpy array,  which means you can apply any known numpy method to it:

# In[ ]:


print(
    "Total number of counts in the image: {:.0f}".format(vela_2fhl.data.sum())
)


# To show the image on the screen we can use the ``plot`` method. It basically calls [plt.imshow](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow), passing the `vela_2fhl.data` attribute but in addition handles axis with world coordinates using [wcsaxes](https://wcsaxes.readthedocs.io/en/latest/) and defines some defaults for nicer plots (e.g. the colormap 'afmhot'):

# In[ ]:


vela_2fhl.plot();


# To make the structures in the image more visible we will smooth the data using a Gausian kernel with a radius of 0.5 deg. Again `smooth()` is a wrapper around existing functionality from the scientific Python libraries. In this case it is Scipy's [gaussian_filter](https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.gaussian_filter.html) method. For convenience the kernel shape can be specified with as string and the smoothing radius with a quantity. It returns again a map object, that we can plot directly the same way we did above:

# In[ ]:


vela_2fhl_smoothed = vela_2fhl.smooth(kernel="gauss", width=0.2 * u.deg)


# In[ ]:


vela_2fhl_smoothed.plot();


# The smoothed plot already looks much nicer, but still the image is rather large. As we are mostly interested in the inner part of the image, we will cut out a quadratic region of the size 9 deg x 9 deg around Vela. Therefore we use ``Map.cutout`` to make a cutout map:

# In[ ]:


# define center and size of the cutout region
center = SkyCoord(265.0, -2.0, unit="deg", frame="galactic")
vela_2fhl_cutout = vela_2fhl_smoothed.cutout(center, 9 * u.deg)
vela_2fhl_cutout.plot();


# To make this exercise a bit more scientifically useful, we will load a second image containing WMAP data from the same region:

# In[ ]:


vela_wmap = Map.read("$GAMMAPY_DATA/images/Vela_region_WMAP_K.fits")

# define a norm to stretch the data, so it is better visible
norm = simple_norm(vela_wmap.data, stretch="sqrt", max_percent=99.9)
vela_wmap.plot(cmap="viridis", norm=norm);


# In order to make the structures in the data better visible we used the [simple_norm()](http://docs.astropy.org/en/stable/api/astropy.visualization.mpl_normalize.simple_norm.html#astropy.visualization.mpl_normalize.simple_norm) method, which allows to stretch the data for plotting. This is very similar to the methods that e.g. `ds9` provides. In addition we used a different colormap called 'viridis'. An overview of different colomaps can be found [here](http://matplotlib.org/examples/color/colormaps_reference.html). 
# 
# Now let's check the details of the WMAP image:

# In[ ]:


vela_wmap


# As you can see it is defined using a tangential projection and ICRS coordinates, which is different from the projection used for the `vela_2fhl` image. To compare both images we have to reproject one image to the WCS of the other. This can be achieved with the ``reproject`` method: 

# In[ ]:


# reproject and cut out the part we're interested in:
vela_wmap_reprojected = vela_wmap.reproject(vela_2fhl.geom)
vela_wmap_reprojected_cutout = vela_wmap_reprojected.cutout(center, 9 * u.deg)
vela_wmap_reprojected_cutout.plot(cmap="viridis", norm=norm);


# Finally we will combine both images in single plot, by plotting WMAP contours on top of the smoothed Fermi-LAT 2FHL image:
# 

# In[ ]:


fig, ax, _ = vela_2fhl_cutout.plot()
ax.contour(vela_wmap_reprojected_cutout.data, cmap="Blues");


# ### Exercises
# 
# * Add a marker and circle at the Vela pulsar position (you can find examples in the WCSAxes [documentation](https://wcsaxes.readthedocs.io/en/latest/overlays.html)).

# ## Event lists
# 
# Almost any high-level gamma-ray data analysis starts with the raw measured counts data, which is stored in event lists. In Gammapy event lists are represented by the [gammapy.data.EventList](https://docs.gammapy.org/0.9/api/gammapy.data.EventList.html) class. 
# 
# In this section we will learn how to:
# 
# * Read event lists from FITS files
# * Access and work with the `EventList` attributes such as `.table` and `.energy` 
# * Filter events lists using convenience methods
# 
# Let's start with the import from the [gammapy.data](https://docs.gammapy.org/0.9/data/index.html) submodule:

# In[ ]:


from gammapy.data import EventList


# Very similar to the sky map class an event list can be created, by passing a filename to the `.read()` method:

# In[ ]:


events_2fhl = EventList.read("$GAMMAPY_DATA/fermi_2fhl/2fhl_events.fits.gz")


# This time the actual data is stored as an [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table) object. It can be accessed with `.table` attribute: 

# In[ ]:


events_2fhl.table


# You can do *len* over event_2fhl.table to find the total number of events.

# In[ ]:


print("Total number of events: {}".format(len(events_2fhl.table)))


# And we can access any other attribute of the `Table` object as well:

# In[ ]:


events_2fhl.table.colnames


# For convenience we can access the most important event parameters as properties on the `EventList` objects. The attributes will return corresponding Astropy objects to represent the data, such as [astropy.units.Quantity](http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html#astropy.units.Quantity), [astropy.coordinates.SkyCoord](http://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) or [astropy.time.Time](http://docs.astropy.org/en/stable/api/astropy.time.Time.html#astropy.time.Time) objects:

# In[ ]:


events_2fhl.energy.to("GeV")


# In[ ]:


events_2fhl.galactic
# events_2fhl.radec


# In[ ]:


events_2fhl.time


# In addition `EventList` provides convenience methods to filter the event lists. One possible use case is to find the highest energy event within a radius of 0.5 deg around the vela position:

# In[ ]:


# select all events within a radius of 0.5 deg around center
events_vela_2fhl = events_2fhl.select_sky_cone(
    center=center, radius=0.5 * u.deg
)

# sort events by energy
events_vela_2fhl.table.sort("ENERGY")

# and show highest energy photon
events_vela_2fhl.energy[-1].to("GeV")


# ### Exercises
# 
# * Make a counts energy spectrum for the galactic center region, within a radius of 10 deg.

# ## Source catalogs
# 
# Gammapy provides a convenient interface to access and work with catalog based data. 
# 
# In this section we will learn how to:
# 
# * Load builtins catalogs from [gammapy.catalog](https://docs.gammapy.org/0.9/catalog/index.html)
# * Sort and index the underlying Astropy tables
# * Access data from individual sources
# 
# Let's start with importing the 2FHL catalog object from the [gammapy.catalog](https://docs.gammapy.org/0.9/catalog/index.html) submodule:

# In[ ]:


from gammapy.catalog import SourceCatalog2FHL


# First we initialize the Fermi-LAT 2FHL catalog and directly take a look at the `.table` attribute:

# In[ ]:


fermi_2fhl = SourceCatalog2FHL(
    "$GAMMAPY_DATA/catalogs/fermi/gll_psch_v08.fit.gz"
)
fermi_2fhl.table


# This looks very familiar again. The data is just stored as an [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table) object. We have all the methods and attributes of the `Table` object available. E.g. we can sort the underlying table by `TS` to find the top 5 most significant sources:
# 
# 

# In[ ]:


# sort table by TS
fermi_2fhl.table.sort("TS")

# invert the order to find the highest values and take the top 5
top_five_TS_2fhl = fermi_2fhl.table[::-1][:5]

# print the top five significant sources with association and source class
top_five_TS_2fhl[["Source_Name", "ASSOC", "CLASS"]]


# If you are interested in the data of an individual source you can access the information from catalog using the name of the source or any alias source name that is defined in the catalog:

# In[ ]:


mkn_421_2fhl = fermi_2fhl["2FHL J1104.4+3812"]

# or use any alias source name that is defined in the catalog
mkn_421_2fhl = fermi_2fhl["Mkn 421"]
print(mkn_421_2fhl.data["TS"])


# ### Exercises
# 
# * Try to load the Fermi-LAT 3FHL catalog and check the total number of sources it contains.
# * Select all the sources from the 2FHL catalog which are contained in the Vela region. Add markers for all these sources and try to add labels with the source names. The function [ax.text()](http://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.text.html#matplotlib.axes.Axes.text) might be helpful.
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


crab_2fhl = fermi_2fhl["Crab"]
print(crab_2fhl.spectral_model)


# The `crab_2fhl.spectral_model` is an instance of the [gammapy.spectrum.models.PowerLaw2](https://docs.gammapy.org/0.9/api/gammapy.spectrum.models.PowerLaw2.html#gammapy.spectrum.models.PowerLaw2) model, with the parameter values and errors taken from the 2FHL catalog. 
# 
# Let's plot the spectral model in the energy range between 50 GeV and 2000 GeV:

# In[ ]:


ax_crab_2fhl = crab_2fhl.spectral_model.plot(
    energy_range=[50, 2000] * u.GeV, energy_power=0
)


# We assign the return axes object to variable called `ax_crab_2fhl`, because we will re-use it later to plot the flux points on top.
# 
# To compute the differential flux at 100 GeV we can simply call the model like normal Python function and convert to the desired units:

# In[ ]:


crab_2fhl.spectral_model(100 * u.GeV).to("cm-2 s-1 GeV-1")


# Next we can compute the integral flux of the Crab between 50 GeV and 2000 GeV:

# In[ ]:


crab_2fhl.spectral_model.integral(emin=50 * u.GeV, emax=2000 * u.GeV).to(
    "cm-2 s-1"
)


# We can easily convince ourself, that it corresponds to the value given in the Fermi-LAT 2FHL catalog:

# In[ ]:


crab_2fhl.data["Flux50"]


# In addition we can compute the energy flux between 50 GeV and 2000 GeV:

# In[ ]:


crab_2fhl.spectral_model.energy_flux(emin=50 * u.GeV, emax=2000 * u.GeV).to(
    "erg cm-2 s-1"
)


# Next we will access the flux points data of the Crab:

# In[ ]:


print(crab_2fhl.flux_points)


# If you want to learn more about the different flux point formats you can read the specification [here](http://gamma-astro-data-formats.readthedocs.io/en/latest/results/flux_points/index.html#flux-points).
# 
# No we can check again the underlying astropy data structure by accessing the `.table` attribute:

# In[ ]:


crab_2fhl.flux_points.table


# Finally let's combine spectral model and flux points in a single plot and scale with `energy_power=2` to obtain the spectral energy distribution:

# In[ ]:


ax = crab_2fhl.spectral_model.plot(
    energy_range=[50, 2000] * u.GeV, energy_power=2
)
crab_2fhl.flux_points.plot(ax=ax, energy_power=2);


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
