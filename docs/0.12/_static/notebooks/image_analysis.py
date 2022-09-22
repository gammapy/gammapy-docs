
# coding: utf-8

# # Fitting 2D images with Gammapy
# 
# Gammapy does not have any special handling for 2D images, but treats them as a subset of maps. Thus, classical 2D image analysis can be done in 2 independent ways: 
# 
# 1. Using the sherpa pacakge, see: [image_fitting_with_sherpa.ipynb](image_fitting_with_sherpa.ipynb),
# 
# 2. Within gammapy, by assuming 2D analysis to be a sub-set of the generalised `maps`. Thus, analysis should proceeexactly as demonstrated in [analysis_3d.ipynb](analysis_3d.ipynb), taking care of a few things that we mention in this tutorial
# 
# We consider 2D `images` to be a special case of 3D `maps`, ie, maps with only one energy bin. This is a major difference while analysing in `sherpa`, where the `maps` must not contain any energy axis. In this tutorial, we do a classical image analysis using three example observations of the Galactic center region with CTA - i.e., study the source flux and morphology.
# 
# 

# ## Setup

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.irf import make_mean_psf
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.cube import MapMaker, PSFKernel, MapDataset
from gammapy.cube.models import SkyModel, BackgroundModel
from gammapy.spectrum.models import PowerLaw2
from gammapy.image.models import SkyPointSource
from gammapy.utils.fitting import Fit


# ## Prepare modeling input data
# 
# ### The counts, exposure and the background maps
# This is the same drill - use `DataStore` object to access the CTA observations and retrieve a list of observations by passing the observations IDs to the `.get_observations()` method, then use `MapMaker` to make the maps.

# In[ ]:


# Define which data to use and print some information
data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
data_store.info()


# In[ ]:


print(
    "Total observation time: {}".format(
        data_store.obs_table["ONTIME"].quantity.sum().to("hour")
    )
)


# In[ ]:


# Select some observations from these dataset by hand
obs_ids = [110380, 111140, 111159]
observations = data_store.get_observations(obs_ids)


# In[ ]:


emin, emax = [0.1, 10] * u.TeV
energy_axis = MapAxis.from_bounds(
    emin.value, emax.value, 10, unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(0, 0),
    binsz=0.02,
    width=(10, 8),
    coordsys="GAL",
    proj="CAR",
    axes=[energy_axis],
)


# Note that even when doing a 2D analysis, it is better to use fine energy bins in the beginning and then sum them over. This is to ensure that the background shape can be approximated by a power law function in each energy bin. The `run_images()` can be used to compute maps in fine bins and then squash them to have one bin. This can be done by specifying `keep_dims = True`. This will compute a summed counts and background maps, and a spectral weighted exposure map.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'maker = MapMaker(geom, offset_max=4.0 * u.deg)\nspectrum = PowerLaw2(index=2)\nmaps2D = maker.run_images(observations, spectrum=spectrum, keepdims=True)')


# In[ ]:


maps2D


# For a typical 2D analysis, using an energy dispersion usually does not make sense. A PSF map can be made as in the regular 3D case, taking care to weight it properly with the spectrum.

# In[ ]:


# mean PSF
geom2d = maps2D["exposure"].geom
src_pos = SkyCoord(0, 0, unit="deg", frame="galactic")
table_psf = make_mean_psf(observations, src_pos)

table_psf_2d = table_psf.table_psf_in_energy_band(
    (emin, emax), spectrum=spectrum
)

# PSF kernel used for the model convolution
psf_kernel = PSFKernel.from_table_psf(
    table_psf_2d, geom2d, max_radius="0.3 deg"
)


# Now, the analysis proceeds as usual. Just take care to use the proper geometry in this case.

# ## Define a mask

# In[ ]:


region = CircleSkyRegion(center=src_pos, radius=0.6 * u.deg)
mask = geom2d.region_mask([region])


# ## Modeling the source
# 
# This is the important thing to note in this analysis. Since modelling and fitting in `gammapy.maps` needs to have a combination of spectral models, we have to use a dummy Powerlaw as for the spectral model and fix its index to 2. Since we are interested only in the integral flux, we will use the `PowerLaw2` model which directly fits an integral flux.

# In[ ]:


spatial_model = SkyPointSource(lon_0="0.01 deg", lat_0="0.01 deg")
spectral_model = PowerLaw2(
    emin=emin, emax=emax, index=2.0, amplitude="3e-12 cm-2 s-1"
)
model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)
model.parameters["index"].frozen = True


# ## Modeling the background
# 
# Gammapy fitting framework assumes the background to be an integrated model.
# Thus, we will define the background as a model, and freeze its parameters for now.

# In[ ]:


background_model = BackgroundModel(maps2D["background"])
background_model.parameters["norm"].frozen = True
background_model.parameters["tilt"].frozen = True


# In[ ]:


dataset = MapDataset(
    model=model,
    counts=maps2D["counts"],
    exposure=maps2D["exposure"],
    background_model=background_model,
    mask_fit=mask,
    psf=psf_kernel,
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit = Fit(dataset)\nresult = fit.run()')


# To see the actual best-fit parameters, do a print on the result

# In[ ]:


print(model)


# In[ ]:


# To get the errors on the model, we can check the covariance table:
result.parameters.covariance_to_table()


# ## Todo: Demonstrate plotting a flux map

# ## Exercises
# 1. Plot residual maps as done in the `analysis_3d` notebook
# 2. Iteratively add and fit sources as explained in `image_fitting_with_sherpa` notebook
# 
