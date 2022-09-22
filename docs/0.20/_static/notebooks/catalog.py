#!/usr/bin/env python
# coding: utf-8

# # Source catalogs
# 
# `~gammapy.catalog` provides convenient access to common gamma-ray source catalogs.
# This module is mostly independent from the rest of Gammapy.
# Typically you use it to compare new analyses against catalog results, e.g. overplot the spectral model, or compare the source position.
# 
# Moreover as creating a source model and flux points for a given catalog from the FITS table is tedious, `~gammapy.catalog` has this already implemented. So you can create initial source models for your analyses.
# This is very common for Fermi-LAT, to start with a catalog model.
# For TeV analysis, especially in crowded Galactic regions, using the HGPS, gamma-cat or 2HWC catalog in this way can also be useful.
# 
# In this tutorial you will learn how to:
# 
# - List available catalogs
# - Load a catalog
# - Access the source catalog table data
# - Select a catalog subset or a single source
# - Get source spectral and spatial models
# - Get flux points (if available)
# - Get lightcurves (if available)
# - Access the source catalog table data
# - Pretty-print the source information
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
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from gammapy.catalog import CATALOG_REGISTRY


# ## List available catalogs
# 
# `~gammapy.catalog` contains a catalog registry ``CATALOG_REGISTRY``, which maps catalog names (e.g. "3fhl") to catalog classes (e.g. ``SourceCatalog3FHL``). 

# In[ ]:


CATALOG_REGISTRY


# ## Load catalogs
# 
# If you have run `gammapy download datasets` or `gammapy download tutorials`,
# you have a copy of the catalogs as FITS files in `$GAMMAPY_DATA/catalogs`,
# and that is the default location where `~gammapy.catalog` loads from.
# 

# In[ ]:


get_ipython().system('ls -1 $GAMMAPY_DATA/catalogs')


# In[ ]:


get_ipython().system('ls -1 $GAMMAPY_DATA/catalogs/fermi')


# So a catalog can be loaded directly from its corresponding class

# In[ ]:


from gammapy.catalog import SourceCatalog4FGL

catalog = SourceCatalog4FGL()
print("Number of sources :", len(catalog.table))


# Note that it loads the default catalog from `$GAMMAPY_DATA/catalogs`, you could pass a different `filename` when creating the catalog.
# For example here we load an older version of 4FGL catalog:  

# In[ ]:


catalog = SourceCatalog4FGL("$GAMMAPY_DATA/catalogs/fermi/gll_psc_v20.fit.gz")
print("Number of sources :", len(catalog.table))


# 
# Alternatively you can load a catalog by name via `CATALOG_REGISTRY.get_cls(name)()` (note the `()` to instantiate a catalog object from the catalog class - only this will load the catalog and be useful), or by importing the catalog class (e.g. `SourceCatalog3FGL`) directly. The two ways are equivalent, the result will be the same.
# 

# In[ ]:


# FITS file is loaded
catalog = CATALOG_REGISTRY.get_cls("3fgl")()
catalog


# In[ ]:


# Let's load the source catalogs we will use throughout this tutorial
catalog_gammacat = CATALOG_REGISTRY.get_cls("gamma-cat")()
catalog_3fhl = CATALOG_REGISTRY.get_cls("3fhl")()
catalog_4fgl = CATALOG_REGISTRY.get_cls("4fgl")()
catalog_hgps = CATALOG_REGISTRY.get_cls("hgps")()


# ## Catalog table
# 
# Source catalogs are given as `FITS` files that contain one or multiple tables.
# 
# However, you can also access the underlying `astropy.table.Table` for a catalog,
# and the row data as a Python `dict`. This can be useful if you want to do something
# that is not pre-scripted by the `~gammapy.catalog` classes, such as e.g. selecting
# sources by sky position or association class, or accessing special source information.
# 

# In[ ]:


type(catalog_3fhl.table)


# In[ ]:


len(catalog_3fhl.table)


# In[ ]:


catalog_3fhl.table[:3][["Source_Name", "RAJ2000", "DEJ2000"]]


# Note that the catalogs object include a helper property that gives directly the sources positions as a `SkyCoord` object (we will show an usage example in the following).

# In[ ]:


catalog_3fhl.positions[:3]


# ## Source object
# 
# ### Select a source
# 
# The catalog entries for a single source are represented by a `SourceCatalogObject`.
# In order to select a source object index into the catalog using `[]`, with a catalog table row index (zero-based, first row is `[0]`), or a source name. If a name is given, catalog table columns with source names and association names ("ASSOC1" in the example below) are searched top to bottom. There is no name resolution web query.
# 

# In[ ]:


source = catalog_4fgl[49]
source


# In[ ]:


source.row_index, source.name


# In[ ]:


source = catalog_4fgl["4FGL J0010.8-2154"]
source.row_index, source.name


# In[ ]:


source.data["ASSOC1"]


# In[ ]:


source = catalog_4fgl["PKS 0008-222"]
source.row_index, source.name


# Note that you can also do a `for source in catalog` loop, to find or process
# sources of interest.
# 
# ###  Source information
# 
# The source objects have a `data` property that contains the information of the catalog row corresponding to the source.

# In[ ]:


source.data["Npred"]


# In[ ]:


source.data["GLON"], source.data["GLAT"]


# As for the catalog object, the source object has a `position` property.

# In[ ]:


source.position.galactic


# ## Select a catalog subset
# 
# The catalog objects support selection using boolean arrays (of the same length), so one can create a new catalog as a subset of the main catalog that verify a set of conditions.
# 
# In the next example we selection only few of the brightest sources brightest sources in the 100 to 200 GeV energy band.

# In[ ]:


mask_bright = np.zeros(len(catalog_3fhl.table), dtype=bool)
for k, source in enumerate(catalog_3fhl):
    flux = (
        source.spectral_model()
        .integral(100 * u.GeV, 200 * u.GeV)
        .to("cm-2 s-1")
    )
    if flux > 1e-10 * u.Unit("cm-2 s-1"):
        mask_bright[k] = True
        print(f"{source.row_index:<7d} {source.name:20s} {flux:.3g}")


# In[ ]:


catalog_3fhl_bright = catalog_3fhl[mask_bright]
catalog_3fhl_bright


# In[ ]:


catalog_3fhl_bright.table["Source_Name"]


# Similarly we can select only sources within a region of interest. Here for example we use the `position` property of the catalog object to select sources whitin 5 degrees from "PKS 0008-222":
# 

# In[ ]:


source = catalog_4fgl["PKS 0008-222"]
mask_roi = source.position.separation(catalog_4fgl.positions) < 5 * u.deg


# In[ ]:


catalog_4fgl_roi = catalog_4fgl[mask_roi]
print("Number of sources :", len(catalog_4fgl_roi.table))


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


energy_bounds = (100 * u.MeV, 100 * u.GeV)
opts = dict(sed_type="e2dnde", yunits=u.Unit("TeV cm-2 s-1"))
model.spectral_model.plot(energy_bounds, **opts)
model.spectral_model.plot_error(energy_bounds, **opts);


# You can create initial source models for your analyses using the `.to_models()` method of the catalog objects. Here for example we create a `Models` object from the 4FGL catalog subset we previously defined:

# In[ ]:


models_4fgl_roi = catalog_4fgl_roi.to_models()
models_4fgl_roi


# ## Specificities of the HGPS catalog
# 
# Using the `.to_models()` method for the `gammapy.catalog.SourceCatalogHGPS` will return only the models components of the sources retained in the main catalog, several candidate objects appears only in the Gaussian components table (see section 4.9 of the HGPS paper, https://arxiv.org/abs/1804.02432). To access these components you can do the following:
# 

# In[ ]:


discarded_ind = np.where(
    [
        "Discarded" in _
        for _ in catalog_hgps.table_components["Component_Class"]
    ]
)[0]
discarded_table = catalog_hgps.table_components[discarded_ind]


# There is no spectral model available for these components but you can access their spatial models:

# In[ ]:


discarded_spatial = [
    catalog_hgps.gaussian_component(idx).spatial_model()
    for idx in discarded_ind
]


# In addition to the source components the HGPS catalog include a large scale diffuse component built by fitting a gaussian model in a sliding window along the Galactic plane. Information on this model can be accessed via the propoerties `.table_large_scale_component` and `.large_scale_component` of `gammapy.catalog.SourceCatalogHGPS`.

# In[ ]:


# here we show the 5 first elements of the table
catalog_hgps.table_large_scale_component[:5]
# you can also try :
# help(catalog_hgps.large_scale_component)


# ## Flux points
# 
# The flux points are available via the `flux_points` property as a `gammapy.spectrum.FluxPoints` object.

# In[ ]:


source = catalog_4fgl["PKS 2155-304"]
flux_points = source.flux_points


# In[ ]:


flux_points


# In[ ]:


flux_points.to_table(sed_type="flux")


# In[ ]:


flux_points.plot(sed_type="e2dnde");


# ## Lightcurves
# 
# The Fermi catalogs contain lightcurves for each source. It is available via the `source.lightcurve()` method as a `~gammapy.estimators.LightCurve` object.

# In[ ]:


lightcurve = catalog_4fgl["4FGL J0349.8-2103"].lightcurve()


# In[ ]:


lightcurve


# In[ ]:


lightcurve.to_table(format="lightcurve", sed_type="flux")


# In[ ]:


lightcurve.plot();


# ## Pretty-print source information
# 
# A source object has a nice string representation that you can print.
# 

# In[ ]:


source = catalog_hgps["MSH 15-52"]
print(source)


# You can also call `source.info()` instead and pass as an option what information to print. The options available depend on the catalog, you can learn about them using `help()`

# In[ ]:


help(source.info)


# In[ ]:


print(source.info("associations"))


# In[ ]:




