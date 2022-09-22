
# coding: utf-8

# # HGPS
# 
# 
# ## Introduction
# 
# The **H.E.S.S. Galactic Plane Survey (HGPS)** is the first deep and wide survey of the Milky Way in TeV gamma-rays.
# 
# In April 2018, a paper on the HGPS was published, and survey maps and a source catalog on the HGPS in FITS format released.
# 
# More information, and the HGPS data for download in FITS format is available here:
# https://www.mpi-hd.mpg.de/hfm/HESS/hgps
# 
# **Please read the Appendix A of the paper to learn about the caveats to using the HGPS data. Especially note that the HGPS survey maps are correlated and thus no detailed source morphology analysis is possible, and also note the caveats concerning spectral models and spectral flux points.**
# 
# ## Notebook Overview
# 
# This is a Jupyter notebook that illustrates how to work with the HGPS data from Python.
# 
# You will learn how to access the HGPS images as well as the HGPS catalog and other tabular data using Astropy and Gammapy.
# 
# * In the first part we will only use Astropy to do some basic things.
# * Then in the second part we'll use Gammapy to do some things that are a little more advanced.
# 
# The notebook is pretty long: feel free to skip ahead to a section of your choice after executing the cells in the "Setup" and "Download data" sections below.
# 
# Note that there are other tools to work with FITS data that we don't explain here. Specifically [DS9](http://ds9.si.edu/) and [Aladin](http://aladin.u-strasbg.fr/) are good FITS image viewers, and [TOPCAT](http://www.star.bris.ac.uk/~mbt/topcat/) is great for FITS tables. Astropy and Gammapy are just one way to work with the HGPS data; any tool that can access FITS data can be used.
# 
# ## Packages
# 
# We will be using the following Python packages
# 
# * [astropy](http://docs.astropy.org/)
# * [gammapy](https://docs.gammapy.org/)
# * [matplotlib](https://matplotlib.org/) for plotting
# 
# Under the hood all of those packages use Numpy arrays to store and work with data.
# 
# More specifically, we will use the following functions and classes:
# 
# * From [astropy](http://docs.astropy.org/), we will use [astropy.io.fits](http://docs.astropy.org/en/stable/io/fits/index.html) to read the FITS data, [astropy.table.Table](http://docs.astropy.org/en/stable/table/index.html) to work with the tables, but also [astropy.coordinates.SkyCoord](http://docs.astropy.org/en/stable/coordinates/index.html) and [astropy.wcs.WCS](http://docs.astropy.org/en/stable/wcs/index.html) to work with sky and pixel coordinates and [astropy.units.Quantity](http://docs.astropy.org/en/stable/units/index.html) to work with quantities.
# 
# * From [gammapy](https://docs.gammapy.org/), we will use [gammapy.maps.WcsNDMap](https://docs.gammapy.org/0.12/api/gammapy.maps.WcsNDMap.html) to work with the HGPS sky maps, and [gammapy.catalog.SourceCatalogHGPS](https://docs.gammapy.org/0.12/api/gammapy.catalog.SourceCatalogHGPS.html) and [gammapy.catalog.SourceCatalogObjectHGPS](https://docs.gammapy.org/0.12/api/gammapy.catalog.SourceCatalogObjectHGPS.html) to work with the HGPS catalog data, especially the HGPS spectral data using [gammapy.spectrum.models.SpectralModel](https://docs.gammapy.org/0.12/api/gammapy.spectrum.models.SpectralModel.html) and [gammapy.spectrum.FluxPoints](https://docs.gammapy.org/0.12/api/gammapy.spectrum.FluxPoints.html) objects.
# 
# * [matplotlib](https://matplotlib.org/) for all plotting. For sky image plotting, we will use matplotlib via [astropy.visualization](http://docs.astropy.org/en/stable/visualization/index.html) and [gammapy.maps.WcsNDMap.plot](https://docs.gammapy.org/0.12/api/gammapy.maps.WcsNDMap.html#gammapy.maps.WcsNDMap.plot).
# 
# If you're not familiar with Python, Numpy, Astropy, Gammapy or matplotlib yet, use the tutorial introductions as explained [here](https://docs.gammapy.org/0.12/tutorials.html), as well as the links to the documentation that we just mentioned.

# ## Setup
# 
# We start by importing everything we will use in this notebook, and configuring the notebook to show plots inline.
# 
# If you get an error here, you probably have to install the missing package and re-start the notebook.
# 
# If you don't get an error, just go ahead, no nead to read the import code and text in this section.

# In[ ]:


import numpy as np


# In[ ]:


import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

import astropy

print(astropy.__version__)


# In[ ]:


from gammapy.maps import Map
from gammapy.image import MapPanelPlotter

from gammapy.catalog import SourceCatalogHGPS

import gammapy

print(gammapy.__version__)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import matplotlib

print(matplotlib.__version__)


# In[ ]:


from urllib.request import urlretrieve
from pathlib import Path


# ## Download Data
# 
# First, you need to download the HGPS FITS data from https://www.mpi-hd.mpg.de/hfm/HESS/hgps .
# 
# If you haven't already, you can use the following commands to download the files to your local working directory.
# 
# You don't have to read the code in the next cell; that's just how to downlaod files from Python.
# You could also download the files with your web browser, or from the command line e.g. with curl:
# 
#     mkdir hgps_data
#     cd hgps_data
#     curl -O https://www.mpi-hd.mpg.de/hfm/HESS/hgps/data/hgps_catalog_v1.fits.gz
#     curl -O https://www.mpi-hd.mpg.de/hfm/HESS/hgps/data/hgps_map_significance_0.1deg_v1.fits.gz
# 
# **The rest of this notebook assumes that you have the data files at ``hgps_data_path``.**

# In[ ]:


# Download HGPS data used in this tutorial to a folder of your choice
# The default `hgps_data` used here is a sub-folder in your current
# working directory (where you started the notebook)
hgps_data_path = Path("hgps_data")

# In this notebook we will only be working with the following files
# so we only download what is needed.
hgps_filenames = [
    "hgps_catalog_v1.fits.gz",
    "hgps_map_significance_0.1deg_v1.fits.gz",
]


# In[ ]:


def hgps_data_download():
    base_url = "https://www.mpi-hd.mpg.de/hfm/HESS/hgps/data/"
    for filename in hgps_filenames:
        url = base_url + filename
        path = hgps_data_path / filename
        if path.exists():
            print("Already downloaded: {}".format(path))
        else:
            print("Downloading {} to {}".format(url, path))
            urlretrieve(url, str(path))


hgps_data_path.mkdir(parents=True, exist_ok=True)
hgps_data_download()

print("\n\nFiles at {} :\n".format(hgps_data_path.absolute()))
for path in hgps_data_path.iterdir():
    print(path)


# ## Catalog with Astropy
# 
# ### FITS file content
# 
# Let's start by just opening up `hgps_catalog_v1.fits.gz` and looking at the content.
# 
# Note that ``astropy.io.fits.open`` doesn't work with `Path` objects yet,
# so you have to call `str(path)` and pass a string.

# In[ ]:


path = hgps_data_path / "hgps_catalog_v1.fits.gz"
hdu_list = fits.open(str(path))


# In[ ]:


hdu_list.info()


# There are six tables. Each table and column is described in detail in the HGPS paper.
# 
# ### Access table data
# 
# We could work with the `astropy.io.fits.HDUList` and `astropy.io.fits.BinTable` objects.
# However, in Astropy a nicer class to work with tables has been developed: `astropy.table.Table`.
# 
# We will only be using `Table`, so let's convert the FITS tabular data into a `Table` object:

# In[ ]:


table = Table.read(hdu_list["HGPS_SOURCES"])

# Alternatively, reading from file directly would work like this:
# table = Table.read(str(path), hdu='HGPS_SOURCES')
# Usually you have to look first what HDUs are in a FITS file
# like we did above; `Table` is just for one table


# In[ ]:


# List available columns
table.info()
# To get shorter output here, we just list a few
# table.info(out=None)['name', 'dtype', 'shape', 'unit'][:10]


# In[ ]:


# Rows are accessed by indexing into the table
# with an integer index (Python starts at index 0)
# table[0]
# To get shorter output here, we just list a few
table[table.colnames[:5]][0]


# In[ ]:


# Columns are accessed by indexing into the table
# with a column name string
# Then you can slice the column to get the rows you want
table["Source_Name"][:5]


# In[ ]:


# Accessing a given element of the table like this
table["Source_Name"][5]


# In[ ]:


# If you know some Python and Numpy, you can now start
# to ask questions about the HGPS data.
# Just to give one example: "What spatial models are used?"


# In[ ]:


set(table["Spatial_Model"])


# ### Convert formats
# 
# The HGPS catalog is only released in FITS format.
# 
# We didn't provide multiple formats (DS9, CSV, XML, VOTABLE, ...) because everyone needs something a little different (in terms of format or content). Instead, here we show how you can convert to any format / content you like wiht a few lines of Python.
# 
# #### DS9 region format
# 
# At [Fermi-LAT 3FGL webpage](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/4yr_catalog/),
# we see that they also provide [DS9 region files](http://ds9.si.edu/doc/ref/region.html) that look like this:
# 
#     global color=red
#     fk5;point(   0.0377, 65.7517)# point=cross text={3FGL J0000.1+6545}
#     fk5;point(   0.0612,-37.6484)# point=cross text={3FGL J0000.2-3738}
#     fk5;point(   0.2535, 63.2440)# point=cross text={3FGL J0001.0+6314}
#     ... more lines, one for each source ...
# 
# because many people use [DS9](http://ds9.si.edu/) to view astronomical images, and to overplot catalog data.
# 
# Let's convert the HGPS catalog into this format.
# (to avoid very long text output, we use `table[:3]` to just use the first three rows)

# In[ ]:


lines = ["global color=red"]
for row in table[:3]:
    fmt = "fk5;point({:.5f},{:.5f})# point=cross text={{{}}}"
    vals = row["RAJ2000"], row["DEJ2000"], row["Source_Name"]
    lines.append(fmt.format(*vals))
txt = "\n".join(lines)


# In[ ]:


# To save it to a DS9 region text file
path = hgps_data_path / "hgps_my_format.reg"

with open(str(path), "w") as f:
    f.write(txt)

# Print content of the file to check
print(path.read_text())


# Note that there is an extra package [astropy-regions](https://astropy-regions.readthedocs.io/) that has classes to represent region objects and supports the DS9 format. We could have used that to generate the DS9 region strings, or we could use it to read the DS9 region file via:
# 
#     import regions
#     region_list = regions.ds9.read_ds9(str(path))
# 
# We will not show examples how to work with regions or how to do region-based analyses here, but if you want to do something like e.g. measure the total flux in a given region in the HGPS maps, the region package would be useful.

# #### CSV format
# 
# Let's do one more example, converting part of the HGPS catalog table information to CSV, i.e. comma-separated format.
# This is a format that can be read by any tool for tabular data, e.g. if you are using Excel or ROOT for your gamma-ray data analysis, this section is for you!
# 
# Usually you can just use the CSV writer in Astropy like this:
# 
#     path = hgps_data_path / 'hgps_my_format.csv'
#     table.write(str(path), format='ascii.csv')
#     
# However, this doesn't work here for two reasons:
# 
# 1. this table contains metadata that trips up the Astropy CSV writer (I filed an [issue](https://github.com/astropy/astropy/issues/7357) with Astropy)
# 1. this table contains array-valued columns for the spectral points and CSV can only have scalar values.

# In[ ]:


# This is the problematic header key
table.meta["comments"]


# In[ ]:


# These are the array-valued columns
array_colnames = tuple(
    name for name in table.colnames if len(table[name].shape) > 1
)
for name in array_colnames:
    print(name, table[name].shape)


# In[ ]:


# So each source has an array of spectral flux points,
# and FITS allows storing such arrays in table cells
# Knowing this, you could work with the spectral points directly.
# We won't give an example here; but instead show how to work
# with HGPS spectra in the Gammapy section below, because it's easier.
print(table["Flux_Points_Energy"][0])
print(table["Flux_Points_Flux"][0])


# Let's get back to our actual goal of converting part of the HGPS catalog table data to CSV format.
# The following code makes a copy of the table that contains just the scalar columns, then removes the `meta` information (most CSV variants / readers don't support metadata anyways) and writes a CSV file.

# In[ ]:


scalar_colnames = tuple(
    name for name in table.colnames if len(table[name].shape) <= 1
)
table2 = table[scalar_colnames]
table2.meta = {}
path = hgps_data_path / "hgps_my_format.csv"
table2.write(str(path), format="ascii.csv", overwrite=True)


# The Astropy ASCII writer and reader supports many variants.
# Let's do one more example, using the ``ascii.fixed_width`` format which is a bit easier to read for humans.
# We will just select a few columns and rows to print here.

# In[ ]:


table2 = table["Source_Name", "GLON", "GLAT"][:3]
table2.meta = {}
table2["Source_Name"].format = "<20s"
table2["GLON"].format = ">8.3f"
table2["GLAT"].format = ">8.3f"
path = hgps_data_path / "hgps_my_format.csv"
table2.write(str(path), format="ascii.fixed_width", overwrite=True)

# Print the CSV file contents to check what we have
print(path.read_text())


# Now you know how to work with the HGPS catalog with Python and Astropy. For the other tables (e.g. `HGPS_GAUSS_COMPONENTS`) it's the same: you should read the description in the HGPS paper, then access the information you need or convert it to the format you want as shows for `HGPS_SOURCES` here). Let's move on and have a look at the HGPS maps.

# ## Maps with Astropy
# 
# This section shows how to load an HGPS survey map with Astropy, and give examples how to work with sky and pixel coordinates to read off map values at given positions.
# 
# We will keep it short, for further examples see the "Maps with Gammapy" section below.
# 
# ### Read
# 
# To read the map, use `fits.open` to get an `HDUList` object, then access `[0]` to get the first and only image HDU in the FITS file, and finally use the `hdu.data` Numpy array, `hdu.header` header object or `wcs = WCS(hdu.header)` WCS object to work with the data.

# In[ ]:


path = hgps_data_path / "hgps_map_significance_0.1deg_v1.fits.gz"
hdu_list = fits.open(str(path))
hdu_list.info()


# In[ ]:


hdu = hdu_list[0]
type(hdu)


# In[ ]:


type(hdu.data)


# In[ ]:


hdu.data.shape


# In[ ]:


# The FITS header contains the information about the
# WCS projection, i.e. the pixel to sky coordinate transform
hdu.header


# ### WCS and coordinates
# 
# To actually do pixel to sky coordinate transformations,
# you have to create a "Word coordinate system transform (WCS)"
# object from the FITS header.

# In[ ]:


wcs = WCS(hdu.header)
wcs


# Let's find the (arguably) most interesting pixel in the HGPS map and look up it's value in the significance image.

# In[ ]:


# pos = SkyCoord.from_name('Sgr A*')
pos = SkyCoord(266.416826, -29.007797, unit="deg")
pos


# In[ ]:


xp, yp = pos.to_pixel(wcs)
xp, yp


# In[ ]:


# Note that SkyCoord makes it easy to transform
# to other frames, and it knows about the frame
# of the WCS object, so for HGPS we have `frame="galactic"`
# and in the call above the transformation from ICRS
# to Galactic and then to pixel all happened in the
# `pos.to_pixel` call. This gives the same pixel postion:
print(pos.galactic)
print(pos.galactic.to_pixel(wcs))


# In[ ]:


# FITS WCS and Numpy have opposite array axis order
# So to look up a pixel in the `hdu.data` Numpy array,
# we need to switch to (y, x), and we also need to
# round correctly to the nearest int, thus the `+ 0.5`
idx = int(yp + 0.5), int(xp + 0.5)
idx


# In[ ]:


# Now, finally the value of this pixel
hdu.data[idx]


# Note that this is a significance map with correlation radius 0.1 deg.
# That means that within a circle of radius 0.1 deg around the pixel center,
# the signal has a significance of 74.2 sigma.
# 
# This does not directly correspond to the significance of a gamma-ray source,
# because this circle contains emission from multiple sources and underlying diffuse emission,
# and for the sources in this circle their emission isn't fully contained because of the size of the HESS PSF.
# 
# We remind you of the caveat the HGPS paper (see Appendix A) that it is not possible to do a detailed quantitative measurement on the released HGPS maps; the measurements in the HGPS paper were done using likelihood fits on uncorrelated counts images taking the PSF shape into account.

# Let's do one more exercise: find the sky position for the pixel with the highest significance:

# In[ ]:


# The pixel with the maximum significance
hdu.data.max()


# In[ ]:


# So it is the pixel in the HGPS map that contains Sgr A*
# and we already roughly know it's position
# Still, let's find the exact pixel center sky position

# We use `np.nanargmax` to find the index, but it's an index
# into the flattened array, so we have to "unravel" it to get a 2D index
yp, xp = np.unravel_index(np.nanargmax(hdu.data), hdu.data.shape)
yp, xp


# In[ ]:


pos = SkyCoord.from_pixel(xp, yp, wcs)
pos


# As you can see, working with FITS images and sky / pixel coordinates directly requires that you learn how to use the WCS object and Numpy arrays and to know that the axis order is `(x, y)` in FITS and WCS, but `(row, column)`, i.e. `(y, x)` in Numpy. It seems quite complex at first, but most astronomers get used to it and manage after a while. `gammapy.maps.WcsNDMap` is a wrapper class that is a bit simpler to use (see below), but under the hood it just calls these Numpy and Astropy methods for you, so it's good to know what is going on in any case.

# ## Catalog with Gammapy
# 
# As you have seen above, working with the HGPS FITS catalog data using Astropy is pretty nice.
# But still, there are some common tasks that aren't trivial to do and require reading the
# FITS table description in detail and writing quite a bit of Python code.
# 
# So that you don't have to, we have done this for HGPS in [gammapy.catalog.SourceCatalogHGPS](https://docs.gammapy.org/0.12/api/gammapy.catalog.SourceCatalogHGPS.html) and also for a few other catalogs that are commonly used in gamma-ray astronomy in [gammapy.catalog](https://docs.gammapy.org/0.12/catalog/index.html).
# 
# ### Read
# 
# Let's start by reading the HGPS catalog via the `SourceCatalogHGPS` class (which is just a wrapper class for `astropy.table.Table`) and access some information about a given source. Feel free to choose any source you like here: we have chosen simply the first one on the table: the pulsar-wind nebula Vela X, a large and bright TeV source around the [Vela pulsar](https://en.wikipedia.org/wiki/Vela_Pulsar).

# In[ ]:


path = hgps_data_path / "hgps_catalog_v1.fits.gz"
cat = SourceCatalogHGPS(path)


# ### Tables
# 
# Now all tables from the FITS file were loaded
# and stored on the ``cat`` object. See the [SourceCatalogHGPS](https://docs.gammapy.org/0.12/api/gammapy.catalog.SourceCatalogHGPS.html) docs, or just try accessing one:

# In[ ]:


cat.table.meta["EXTNAME"]


# In[ ]:


cat.table_components.meta["EXTNAME"]


# ### Source
# 
# You can access a given source by row index (starting at zero) or by source name.
# This creates [SourceCatalogObjectHGPS](https://docs.gammapy.org/0.12/api/gammapy.catalog.SourceCatalogObjectHGPS.html) objects that have a copy of all the data for a given source. See the class docs for a full overview.

# In[ ]:


# These all give the same source object
source = cat[0]
# When accessing by name, the value has to match exactly
# the content from one of these columns:
# `Source_Name` or `Identified_Object`
# which in this case are "HESS J0835-455" and "Vela X"
source = cat["HESS J0835-455"]
source = cat["Vela X"]
source


# In[ ]:


# To see a pretty-printed text version of all
# HGPS data on a given source:
print(source)


# In[ ]:


# You can also more selectively print subsets of info:
print(source.info("map"))


# In[ ]:


# All of the data for this source is available
# via the `source.data` dictionary if you want
# to do some computations
source.data["Flux_Spec_Int_1TeV"]


# In[ ]:


# The flux points are available as a Table
source.flux_points.table


# In[ ]:


# The spectral model is available as a
# Gammapy spectral model object:
spectral_model = source.spectral_model()
print(spectral_model)


# In[ ]:


# One common task is to compute integral fluxes
# The error is computed using the covariance matrix
# (off-diagonal info not given in HGPS, i.e. this is an approximation)
spectral_model.integral_error(emin=1 * u.TeV, emax=10 * u.TeV)


# In[ ]:


# Let's plot the spectrum
source.spectral_model().plot(source.energy_range)
source.spectral_model().plot_error(source.energy_range)
source.flux_points.plot();


# In[ ]:


# Or let's make the same plot in the common
# format with y = E^2 * dnde
opts = dict(energy_power=2, flux_unit="erg-1 cm-2 s-1")
source.spectral_model().plot(source.energy_range, **opts)
source.spectral_model().plot_error(source.energy_range, **opts)
source.flux_points.plot(**opts)
plt.ylabel("E^2 dN/dE (erg cm-2 s-1)")
plt.title("Vela X HGPS spectrum");


# In the next section we will see how to work with the HGPS survey maps from Gammapy, as well as work with other data from the catalog (position and morphology information).

# ## Maps with Gammapy
# 
# Let's use the [gammapy.maps.Map.read](https://docs.gammapy.org/0.12/api/gammapy.maps.Map.html#gammapy.maps.Map.read) method to load up the HGPS significance survey map.

# In[ ]:


path = hgps_data_path / "hgps_map_significance_0.1deg_v1.fits.gz"
survey_map = Map.read(path)


# In[ ]:


# Map has a quick-look plot method, but it's not
# very useful for a survey map that wide with default settings
survey_map.plot();


# In[ ]:


# This is a little better
fig = plt.figure(figsize=(15, 3))
_ = survey_map.plot(stretch="sqrt")
# Note that we also assign the return value (a tuple)
# from the plot method call to a variable called `_`
# This is to avoid Jupyter printing it like in the last cell,
# and generally `_` is a variable name used in Python
# for things you don't want to name or care about at all


# Let's look at a cutout of the Galactic center:

# In[ ]:


image = survey_map.cutout(pos, width=(3.8, 2.5) * u.deg)
fig, ax, _ = image.plot(stretch="sqrt", cmap="inferno")
ax.coords[0].set_major_formatter("dd")
ax.coords[1].set_major_formatter("dd")


# Side comment: If you like, you can format stuff to make it a bit more pretty. With a few lines you can get nice plots, with a few dozen publication-quality images. This is using [matplotlib](https://matplotlib.org/) and [astropy.visualization](http://docs.astropy.org/en/stable/visualization/index.html). Both are pretty complex, but there's many examples available and there's not really another good alternative anyways for astronomical sky images at the moment, so you should just go ahead and learn those.
# 
# There's also a convenience method to look up the map value at a given sky position.
# Let's repeat this for the same position we did above with Numpy and Astropy:

# In[ ]:


pos = SkyCoord(266.416826, -29.007797, unit="deg")
survey_map.get_by_coord(pos)


# ### Vela X
# 
# To finish up this tutorial, let's do something a bit more advanced, than involves the survey map, the HGPS source catalog, the multi-Gauss morphology component model. Let's show the Vela X pulsar wind nebula and the Vela Junior supernova remnant, and overplot some HGPS catalog data.

# In[ ]:


# The spatial model for Vela X in HGPS was three Gaussians
print(source.name)
for component in source.components:
    print(component)


# In[ ]:


from astropy.visualization import simple_norm
from matplotlib.patches import Circle

# Cutout and plot a nice image
pos = SkyCoord(264.5, -2.5, unit="deg", frame="galactic")
image = survey_map.cutout(pos, width=("6 deg", "4 deg"))
norm = simple_norm(image.data, stretch="sqrt", min_cut=0, max_cut=20)
fig = plt.figure(figsize=(12, 8))
fig, ax, _ = image.plot(fig=fig, norm=norm, cmap="inferno")
transform = ax.get_transform("galactic")

# Overplot the pulsar
# print(SkyCoord.from_name('Vela pulsar').galactic)
ax.scatter(263.551, -2.787, transform=transform, s=500, color="cyan")

# Overplot the circle that was used for the HGPS spectral measurement of Vela X
# It is centered on the centroid of the emission and has a radius of 0.5 deg
x = source.data["GLON"].value
y = source.data["GLAT"].value
r = source.data["RSpec"].value
c = Circle(
    (x, y), r, transform=transform, edgecolor="white", facecolor="none", lw=4
)
ax.add_patch(c)

# Overplot circles that represent the components
for c in source.components:
    x = c.data["GLON"].value
    y = c.data["GLAT"].value
    r = c.data["Size"].value
    c = Circle(
        (x, y), r, transform=transform, edgecolor="0.7", facecolor="none", lw=3
    )
    ax.add_patch(c)


# We note that for HGPS there are already spatial models available:
# 
#     print(source.spatial_model())
#     source.components[0].spatial_model
# 
# With some effort, you can use those to make HGPS model flux images.
# 
# There are two reasons we're not showing this here: First, the spatial model code in Gammapy is work in progress, it will change soon. Secondly, doing morphology measurements on the public HGPS maps is discouraged; we note again that the maps are correlated and no detailed PSF information is published. So please be careful / conservative when extracting quantitative mesurements from the HGPS maps e.g. for a source of interest for you.

# ### Survey Map Panel Plot
# 
# The survey maps have an aspect ratio of ~18:2 which makes it hard to fit them on a standard size a4 paper or slide for showing them. In Gammapy there is a helper class `MapPanelPlotter`, which allows to plot these kind of maps on mutiple panels. Here is an example, how to uses it:  

# In[ ]:


fig = plt.figure(figsize=(15, 8))

xlim = Angle([65, 255], unit="deg")
ylim = Angle([-4, 4], unit="deg")

plotter = MapPanelPlotter(
    figure=fig,
    xlim=xlim,
    ylim=ylim,
    npanels=4,
    top=0.98,
    bottom=0.07,
    right=0.98,
    left=0.05,
    hspace=0.15,
)

axes = plotter.plot(
    survey_map, cmap="inferno", stretch="sqrt", vmin=0, vmax=50
)


# Internally the class uses matplotlib class [`GridSpec`](https://matplotlib.org/api/gridspec_api.html#matplotlib.gridspec.GridSpec) to set up a grid of subplot axes, which are returned by the `MapPanelPlotter.plot()` function. These can be used further, e.g. to plot markers on, using the standard [`WcsAxes`](http://docs.astropy.org/en/stable/visualization/wcsaxes/) api.

# ## Conclusions
# 
# This concludes this tutorial how to access and work with the HGPS data from Python, using Astropy and Gammapy.
# 
# 
# * If you have any questions about the HGPS data, please use the contact given at https://www.mpi-hd.mpg.de/hfm/HESS/hgps/ .
# * If you have any questions or issues about Astropy or Gammapy, please use the Gammapy mailing list (see https://gammapy.org/contact.html).
# 
# **Please read the Appendix A of the paper to learn about the caveats to using the HGPS data. Especially note that the HGPS survey maps are correlated and thus no detailed source morphology analysis is possible, and also note the caveats concerning spectral models and spectral flux points.**

# ## Exercises
# 
# * Re-run this notebook, but change the HGPS source that was used for examples. If you have any questions about the data, try to access and print or plot it. Just to give a few ideas: How many identified pulsar wind nebulae are in HGPS? Which are the 5 brightest HGPS sources in the 1-10 TeV energy band? Which is the second-highest significance source in the HPGS image after the Galactic center?
# * Try to reproduce some of the figures in the HGPS paper. Don't try to reproduce them exactly, but just try to write ~ 10 lines of code to access the relevant HGPS data and make a quick plot that shows the same or similar information.
# * Fit a spectral model to the spectral points of Vela X and compare with the HGPS model fit. (they should be similar, but not identical, in HGPS a likelihood fit to counts data was done). For this task, the [sed_fitting_gammacat_fermi](sed_fitting_gammacat_fermi.ipynb) tutorials will be useful.
# 
