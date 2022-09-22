#!/usr/bin/env python
# coding: utf-8

# # Source catalogs
# 
# `~gammapy.catalog` provides convenient access to common gamma-ray source catalogs. E.g. creating a spectral model and spectral points for a given Fermi-LAT catalog and source from the FITS table is tedious, `~gammapy.catalog` has this implemented and makes it easy.
# 
# In this tutorial you will learn how to:
# 
# - List available catalogs
# - Load a catalog
# - Select a source
# - Pretty-print the source information
# - Get source spectral and spatial models
# - Get flux points (if available)
# - Get lightcurves (if available)
# - Access the source catalog table data
# 
# In this tutorial we will show examples using the following catalogs:
# 
# - `~gammapy.catalog.SourceCatalogHGPS`
# - `~gammapy.catalog.SourceCatalogGammaCat`
# - `~gammapy.catalog.SourceCatalog3FHL`
# - `~gammapy.catalog.SourceCatalog4FGL`
# 
# All catalog and source classes work the same, as long as some information is available. E.g. trying to access a lightcurve from a catalog and source that doesn't have that information will return ``None``.
# 
# Further information is available at `~gammapy.catalog`.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


import astropy.units as u
from gammapy.catalog import CATALOG_REGISTRY


# ## List available catalogs
# 
# `~gammapy.catalog` contains a catalog registry ``CATALOG_REGISTRY``, which maps catalog names (e.g. "3fhl") to catalog classes (e.g. ``SourceCatalog3FHL``). 

# In[ ]:


CATALOG_REGISTRY


# In[ ]:


list(CATALOG_REGISTRY)


# ## Load catalogs
# 
# If you have run `gammapy download datasets` or `gammapy download tutorials`,
# you have a copy of the catalogs as FITS files in `$GAMMAPY_DATA/catalogs`,
# and that is the default location where `~gammapy.catalog` loads from.
# 
# You can load a catalog by name via `CATALOG_REGISTRY.get_cls(name)()` (note the `()` to instantiate a catalog object from the catalog class - only this will load the catalog and be useful), or by importing the catalog class (e.g. `SourceCatalog3FGL`) directly. The two ways are equivalent, the result will be the same.
# 
# Note that `$GAMMAPY_DATA/catalogs` is just the default, you could pass a different `filename` when creating the catalog.

# In[ ]:


get_ipython().system('ls -1 $GAMMAPY_DATA/catalogs')


# In[ ]:


# Catalog object - FITS file is loaded
catalog = CATALOG_REGISTRY.get_cls("3fgl")()
catalog


# In[ ]:


from gammapy.catalog import SourceCatalog3FGL

catalog = SourceCatalog3FGL()
catalog


# In[ ]:


# Let's load the source catalogs we will use throughout this tutorial
catalog_gammacat = CATALOG_REGISTRY.get_cls("gamma-cat")()
catalog_3fhl = CATALOG_REGISTRY.get_cls("3fhl")()
catalog_4fgl = CATALOG_REGISTRY.get_cls("4fgl")()
catalog_hgps = CATALOG_REGISTRY.get_cls("hgps")()


# ## Select a source
# 
# To create a source object, index into the catalog using `[]`, passing a catalog table row index (zero-based, first row is `[0]`), or a source name. If passing a name, catalog table columns with source names and association names ("ASSOC1" in the example below) are searched top to bottom. There is no name resolution web query.

# In[ ]:


source = catalog_4fgl[42]
source


# In[ ]:


source.row_index, source.name


# In[ ]:


source = catalog_4fgl["4FGL J0010.8-2154"]
source


# In[ ]:


source.row_index, source.name


# In[ ]:


source.data["ASSOC1"]


# In[ ]:


source = catalog_4fgl["PKS 0008-222"]
source.row_index, source.name


# ## Pretty-print source information
# 
# A source object has a nice string representation that you can print.
# You can also call `source.info()` instead and pass an option what information to print.

# In[ ]:


source = catalog_hgps["MSH 15-52"]
print(source)


# In[ ]:


print(source.info("associations"))


# ## Source models
# 
# The `~gammapy.catalog.SourceCatalogObject` classes have a `sky_model()` model
# which creates a `gammapy.modeling.models.SkyModel` object, with model parameter
# values and parameter errors from the catalog filled in.
# 
# In most cases, the `spectral_model()` method provides the `gammapy.modeling.models.SpectralModel`
# part of the sky model, and the `spatial_model()` method the `gammapy.modeling.models.SpatialModel`
# part individually.
# 
# We use the `gammapy.catalog.SourceCatalog3FHL` for the examples in this section.

# In[ ]:


source = catalog_4fgl["PKS 2155-304"]


# In[ ]:


model = source.sky_model()
model


# In[ ]:


print(model)


# In[ ]:


print(model.spatial_model)


# In[ ]:


print(model.spectral_model)


# In[ ]:


energy_range = (100 * u.MeV, 100 * u.GeV)
opts = dict(energy_power=2, flux_unit="erg-1 cm-2 s-1")
model.spectral_model.plot(energy_range, **opts)
model.spectral_model.plot_error(energy_range, **opts)


# ## Flux points
# 
# The flux points are available via the `flux_points` property as a `gammapy.spectrum.FluxPoints` object.

# In[ ]:


source = catalog_4fgl["PKS 2155-304"]
flux_points = source.flux_points


# In[ ]:


flux_points


# In[ ]:


flux_points.table[["e_min", "e_max", "flux", "flux_errn"]]


# In[ ]:


flux_points.plot()


# ## Lightcurves
# 
# The Fermi catalogs contain lightcurves for each source. It is available via the `source.lightcurve` property as a `~gammapy.time.LightCurve` object.

# In[ ]:


lightcurve = catalog_4fgl["4FGL J0349.8-2103"].lightcurve


# In[ ]:


lightcurve


# In[ ]:


lightcurve.table[:3]


# In[ ]:


lightcurve.plot()


# ## Catalog table and source dictionary
# 
# Source catalogs are given as `FITS` files that contain one or multiple tables.
# Above we showed how to get spectra, light curves and other information as Gammapy objects.
# 
# However, you can also access the underlying `astropy.table.Table` for a catalog,
# and the row data as a Python `dict`. This can be useful if you want to do something
# that is not pre-scripted by the `~gammapy.catalog` classes, such as e.g. selecting
# sources by sky position or association class, or accessing special source information
# (like e.g. `Npred` in the example below).
# 
# Note that you can also do a `for source in catalog` loop, to find or process
# sources of interest.

# In[ ]:


type(catalog_3fhl.table)


# In[ ]:


len(catalog_3fhl.table)


# In[ ]:


catalog_3fhl.table[:3][["Source_Name", "RAJ2000", "DEJ2000"]]


# In[ ]:


source = catalog_3fhl["PKS 2155-304"]


# In[ ]:


source.data["Source_Name"]


# In[ ]:


source.data["Npred"]


# In[ ]:


source.position


# In[ ]:


# Find the brightest sources in the 100 to 200 GeV energy band
for source in catalog_3fhl:
    flux = (
        source.spectral_model()
        .integral(100 * u.GeV, 200 * u.GeV)
        .to("cm-2 s-1")
    )
    if flux > 1e-10 * u.Unit("cm-2 s-1"):
        print(f"{source.row_index:<7d} {source.name:20s} {flux:.3g}")


# ## Exercises
# 
# - How many sources are in the 4FGL catalog? (try `len(catalog.table)`
# - What is the name of the source with row index 42?
# - What is the row index of the source with name "4FGL J0536.1-1205"?
# - What is the integral flux of "4FGL J0536.1-1205" in the energy range 100 GeV to 1 TeV according to the best-fit spectral model?
# - Which source in the HGPS catalog is closest to Galactic position `glon = 42 deg` and `glat = 0 deg`?

# In[ ]:


# Start coding here ...


# ## Next steps
# 
# `~gammapy.catalog` is mostly independent from the rest of Gammapy.
# Typically you use it to compare new analyses against catalog results, e.g. overplot the spectral model, or compare the source position.
# 
# You can also use `~gammapy.catalog` in your scripts to create initial source models for your analyses.
# This is very common for Fermi-LAT, to start with a catalog model.
# For TeV analysis, especially in crowded Galactic regions, using the HGPS, gamma-cat or 2HWC catalog in this way can also be useful.
# 
