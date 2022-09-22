#!/usr/bin/env python
# coding: utf-8

# # Gammapy Models
# 
# 
# This is an introduction and overview on how to work with models in Gammapy. 
# 
# The sub-package `~gammapy.modeling` contains all the functionality related to modeling and fitting
# data. This includes spectral, spatial and temporal model classes, as well as the fit
# and parameter API. We will cover the follwing topics in order:
# 
# 1. [Spectral Models](#Spectral-Models)
# 1. [Spatial Models](#Spatial-Models)
# 1. [SkyModel](#SkyModel)
# 1. [Model Lists and Serialisation](#Model-Lists-and-Serialisation)
# 1. [Implementing as Custom Model](#Implementing-a-Custom-Model)
# 
# The models follow a naming scheme which contains the category as a suffix to the class name. An overview of all the available models can be found in the [model gallery](https://docs.gammapy.org/dev/modeling/gallery/index.html#spectral-models).
# 
# Note that there is a separate tutorial [modeling](modeling.ipynb) that explains about `~gammapy.modeling`,
# the Gammapy modeling and fitting framework. You have to read that to learn how to work with models in order to analyse data.
# 
# 

# # Setup

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


from astropy import units as u
from gammapy.maps import Map, WcsGeom, MapAxis


# # Spectral Models
# 
# All models are imported from the `~gammapy.modeling.models` namespace. Let's start with a `PowerLawSpectralModel`:

# In[ ]:


from gammapy.modeling.models import PowerLawSpectralModel


# In[ ]:


pwl = PowerLawSpectralModel()
print(pwl)


# To get a list of all available spectral models you can import and print the spectral model registry or take a look at the [model gallery](https://docs.gammapy.org/dev/modeling/gallery/index.html#spectral-models):

# In[ ]:


from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY

print(SPECTRAL_MODEL_REGISTRY)


# Spectral models all come with default parameters. Different parameter
# values can be passed on creation of the model, either as a string defining
# the value and unit or as an `astropy.units.Quantity` object directly:

# In[ ]:


amplitude = 1e-12 * u.Unit("TeV-1 cm-2 s-1")
pwl = PowerLawSpectralModel(amplitude=amplitude, index=2.2)


# For convenience a `str` specifying the value and unit can be passed as well:

# In[ ]:


pwl = PowerLawSpectralModel(amplitude="2.7e-12 TeV-1 cm-2 s-1", index=2.2)
print(pwl)


# The model can be evaluated at given energies by calling the model instance:

# In[ ]:


energy = [1, 3, 10, 30] * u.TeV
dnde = pwl(energy)
print(dnde)


# The returned quantity is a differential photon flux. 
# 
# For spectral models you can computed in addition the integrated and energy flux
# in a given energy range:

# In[ ]:


flux = pwl.integral(energy_min=1 * u.TeV, energy_max=10 * u.TeV)
print(flux)

eflux = pwl.energy_flux(energy_min=1 * u.TeV, energy_max=10 * u.TeV)
print(eflux)


# This also works for a list or an array of integration boundaries:

# In[ ]:


energy = [1, 3, 10, 30] * u.TeV
flux = pwl.integral(energy_min=energy[:-1], energy_max=energy[1:])
print(flux)


# In some cases it can be useful to find use the inverse of a spectral model, to find the energy at which a given flux is reached:

# In[ ]:


dnde = 2.7e-12 * u.Unit("TeV-1 cm-2 s-1")
energy = pwl.inverse(dnde)
print(energy)


# As a convenience you can also plot any spectral model in a given energy range:

# In[ ]:


pwl.plot(energy_range=[1, 100] * u.TeV)


# # Spatial Models

# Spatial models are imported from the same `~gammapy.modeling.models` namespace, let's start with a `GaussianSpatialModel`:

# In[ ]:


from gammapy.modeling.models import GaussianSpatialModel


# In[ ]:


gauss = GaussianSpatialModel(lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg")
print(gauss)


# Again you can check the `SPATIAL_MODELS` registry to see which models are available or take a look at the [model gallery](https://docs.gammapy.org/dev/modeling/gallery/index.html#spatial-models).

# In[ ]:


from gammapy.modeling.models import SPATIAL_MODEL_REGISTRY

print(SPATIAL_MODEL_REGISTRY)


# The default coordinate frame for all spatial models is ``"icrs"``, but the frame can be modified using the
# ``frame`` argument:

# In[ ]:


gauss = GaussianSpatialModel(
    lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg", frame="galactic"
)


# You can specify any valid `astropy.coordinates` frame. The center position of the model can be retrieved as a `astropy.coordinates.SkyCoord` object using `SpatialModel.position`: 

# In[ ]:


print(gauss.position)


# Spatial models can be evaluated again by calling the instance:

# In[ ]:


lon = [0, 0.1] * u.deg
lat = [0, 0.1] * u.deg

flux_per_omega = gauss(lon, lat)
print(flux_per_omega)


# The returned quantity corresponds to a surface brightness. Spatial model
# can be also evaluated using `~gammapy.maps.Map` and `~gammapy.maps.Geom` objects:

# In[ ]:


m = Map.create(skydir=(0, 0), width=(1, 1), binsz=0.02, frame="galactic")
m.quantity = gauss.evaluate_geom(m.geom)
m.plot(add_cbar=True);


# Again for convenience the model can be plotted directly:

# In[ ]:


gauss.plot(add_cbar=True);


# All spatial models have an associated sky region to it e.g. to illustrate the extend of the model on a sky image. The returned object is an `regions.SkyRegion` object:

# In[ ]:


print(gauss.to_region())


# Now we can plot the region on an sky image:

# In[ ]:


# create and plot the model
gauss_elongated = GaussianSpatialModel(
    lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg", e=0.7, phi="45 deg"
)
ax = gauss_elongated.plot(add_cbar=True)

# add region illustration
region = gauss_elongated.to_region()
region_pix = region.to_pixel(ax.wcs)
ax.add_artist(region_pix.as_artist());


# The `.to_region()` method can also be useful to write e.g. ds9 region files using `write_ds9` from the `regions` package:

# In[ ]:


from regions import write_ds9

regions = [gauss.to_region(), gauss_elongated.to_region()]

filename = "regions.reg"
write_ds9(regions, filename, coordsys="galactic", fmt=".4f", radunit="deg")


# In[ ]:


get_ipython().system('cat regions.reg')


# # SkyModel

# The `~gammapy.modeling.models.SkyModel` class combines a spectral and a spatial model. It can be created
# from existing spatial and spectral model components:

# In[ ]:


from gammapy.modeling.models import SkyModel

model = SkyModel(spectral_model=pwl, spatial_model=gauss, name="my-source")
print(model)


# It is good practice to specify a name for your sky model, so that you can access it later by name and have meaningful identifier you serilisation. If you don't define a name, a unique random name is generated:

# In[ ]:


model_without_name = SkyModel(spectral_model=pwl, spatial_model=gauss)
print(model_without_name.name)


# The spectral and spatial component of the source model can be accessed using `.spectral_model` and `.spatial_model`:

# In[ ]:


model.spectral_model


# In[ ]:


model.spatial_model


# And can be used as you have seen already seen above:

# In[ ]:


model.spectral_model.plot(energy_range=[1, 10] * u.TeV);


# In some cases (e.g. when doing a spectral analysis) there is only a spectral model associated with the source. So the spatial model is optional:

# In[ ]:


model_spectrum = SkyModel(spectral_model=pwl, name="source-spectrum")
print(model_spectrum)


# Additionally the spatial model of `~gammapy.modeling.models.SkyModel` can be used to represent source models based on templates, where the spatial and energy axes are correlated. It can be created e.g. from an existing FITS file:
# 
# 

# In[ ]:


from gammapy.modeling.models import TemplateSpatialModel
from gammapy.modeling.models import PowerLawNormSpectralModel


# In[ ]:


diffuse_cube = TemplateSpatialModel.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz", normalize=False
)
diffuse = SkyModel(PowerLawNormSpectralModel(), diffuse_cube)
print(diffuse)


# Note that if the spatial model is not normalized over the sky it has to be combined with a normalized spectral model, for example `~gammapy.modeling.models.PowerLawNormSpectralModel`. This is the only case in `gammapy.models.SkyModel` where the unit is fully attached to the spatial model.

# # Model Lists and Serialisation
# 
# In a typical analysis scenario a model consists of mutiple model components, or a "catalog" or "source library". To handle this list of multiple model components, Gammapy has a `Models` class:

# In[ ]:


from gammapy.modeling.models import Models


# In[ ]:


models = Models([model, diffuse])
print(models)


# Individual model components in the list can be accessed by their name:

# In[ ]:


print(models["my-source"])


# **Note:**To make the access by name unambiguous, models are required to have a unique name, otherwise an error will be thrown.
# 
# To see which models are available you can use the `.names` attribute:

# In[ ]:


print(models.names)


# Note that a `SkyModel` object can be evaluated for a given longitude, latitude, and energy, but the `Models` object cannot. This `Models` container object will be assigned to `Dataset` or `Datasets` together with the data to be fitted as explained in other analysis tutorials (see for example the [modeling](modeling.ipynb) notebook).
# 
# The `Models` class also has in place `.append()` and `.extend()` methods:

# In[ ]:


model_copy = model.copy(name="my-source-copy")
models.append(model_copy)


# This list of models can be also serialised to a custom YAML based format: 

# In[ ]:


models_yaml = models.to_yaml()
print(models_yaml)


# The structure of the yaml files follows the structure of the python objects.
# The `components` listed correspond to the `SkyModel` and `SkyDiffuseCube` components of the `Models`. 
# For each `SkyModel` we have  informations about its `name`, `type` (corresponding to the tag attribute) and sub-mobels (i.e `spectral` model and eventually `spatial` model). Then the spatial and spectral models are defiend by their type and parameters. The `parameters` keys name/value/unit are mandatory, while the keys min/max/frozen are optionnals (so you can prepare shorter files).
# 
# If you want to write this list of models to disk and read it back later you can use:

# In[ ]:


models.write("models.yaml", overwrite=True)


# In[ ]:


models_read = Models.read("models.yaml")


# Additionally the models can exported and imported togeter with the data using the `Datasets.read()` and `Datasets.write()` methods as shown in the [analysis_mwl](analysis_mwl.ipynb) notebook.

# # Implementing a Custom Model
# 
# In order to add a user defined spectral model you have to create a SpectralModel subclass.
# This new model class should include:
# 
# - a tag used for serialization (it can be the same as the class name)
# - an instantiation of each Parameter with their unit, default values and frozen status
# - the evaluate function where the mathematical expression for the model is defined.
# 
# As an example we will use a PowerLawSpectralModel plus a Gaussian (with fixed width).
# First we define the new custom model class that we name `MyCustomSpectralModel`:

# In[ ]:


from gammapy.modeling.models import SpectralModel, Parameter


class MyCustomSpectralModel(SpectralModel):
    """My custom spectral model, parametrising a power law plus a Gaussian spectral line.
    
    Parameters
    ----------
    amplitude : `astropy.units.Quantity`
        Amplitude of the spectra model.
    index : `astropy.units.Quantity`
        Spectral index of the model.
    reference : `astropy.units.Quantity`
        Reference energy of the power law.
    mean : `astropy.units.Quantity`
        Mean value of the Gaussian.
    width : `astropy.units.Quantity`
        Sigma width of the Gaussian line.
    
    """

    tag = "MyCustomSpectralModel"
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1", min=0)
    index = Parameter("index", 2, min=0)
    reference = Parameter("reference", "1 TeV", frozen=True)
    mean = Parameter("mean", "1 TeV", min=0)
    width = Parameter("width", "0.1 TeV", min=0, frozen=True)

    @staticmethod
    def evaluate(energy, index, amplitude, reference, mean, width):
        pwl = PowerLawSpectralModel.evaluate(
            energy=energy,
            index=index,
            amplitude=amplitude,
            reference=reference,
        )
        gauss = amplitude * np.exp(-((energy - mean) ** 2) / (2 * width ** 2))
        return pwl + gauss


# It is good practice to also implement a docstring for the model, defining the parameters and also definig a `tag`, which specifies the name of the model for serialisation. Also note that gammapy assumes that all SpectralModel evaluate functions return a flux in unit of `"cm-2 s-1 TeV-1"` (or equivalent dimensions).
# 
# 
# 
# This model can now be used as any other spectral model in Gammapy:

# In[ ]:


my_custom_model = MyCustomSpectralModel(mean="3 TeV")
print(my_custom_model)


# In[ ]:


my_custom_model.integral(1 * u.TeV, 10 * u.TeV)


# In[ ]:


my_custom_model.plot(energy_range=[1, 10] * u.TeV)


# As a next step we can also register the custom model in the `SPECTRAL_MODELS` registry, so that it becomes available for serilisation:

# In[ ]:


SPECTRAL_MODEL_REGISTRY.append(MyCustomSpectralModel)


# In[ ]:


model = SkyModel(spectral_model=my_custom_model, name="my-source")
models = Models([model])
models.write("my-custom-models.yaml", overwrite=True)


# In[ ]:


get_ipython().system('cat my-custom-models.yaml')


# Similarly you can also create custom spatial models and add them to the `SPATIAL_MODELS` registry. In that case gammapy assumes that the evaluate function return a normalized quantity in "sr-1" such as the model integral over the whole sky is one.

# # Models with Energy dependent morphology
# 
# A common science case in the study of extended sources is to probe for energy dependent morphology, in Supernova Remnants or Pulsar Wind Nebulae. Traditionally, this has been done by splitting the data into energy bands and doing individual fits of the morphology in these energy bands.
# 
# `SkyModel` offers a natural framework to simultaneously model the energy and morphology, e.g. spatial extent described by a parametric model expression with energy dependent parameters.
# 
# The models shipped within gammapy use a “factorised” representation of the source model, where the spatial ($l,b$), energy ($E$) and time ($t$) dependence are independent model components and not correlated:
# 
#    $$f(l, b, E, t) = F(l, b) \cdot G(E) \cdot H(t) $$
#     
# To use full 3D models, ie $f(l, b, E) = F(l, b, E) \cdot G(E) $,  you have to implement your own custom `SpatialModel`. Note that it is still necessary to multiply by a `SpectralModel`, $G(E)$ to be dimensionally consistent.
# 
# In this example, we create Gaussian Spatial Model with the extension varying with energy. For simplicity, we assume a linear dependence on energy and parameterize this by specifing the extension at 2 energies. You can add more complex dependences, probably motivated by physical models.

# In[ ]:


from gammapy.modeling.models import SpatialModel
from astropy.coordinates.angle_utilities import angular_separation


class MyCustomGaussianModel(SpatialModel):
    """My custom Energy Dependent Gaussian model.

    Parameters
    ----------
    lon_0, lat_0 : `~astropy.coordinates.Angle`
        Center position
    sigma_1TeV : `~astropy.coordinates.Angle`
        Width of the Gaussian at 1 TeV
    sigma_10TeV : `~astropy.coordinates.Angle`
        Width of the Gaussian at 10 TeV

    """

    tag = "MyCustomGaussianModel"
    lon_0 = Parameter("lon_0", "0 deg")
    lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)

    sigma_1TeV = Parameter("sigma_1TeV", "2.0 deg", min=0)
    sigma_10TeV = Parameter("sigma_10TeV", "0.2 deg", min=0)

    @staticmethod
    def get_sigma(energy, sigma_1TeV, sigma_10TeV):
        """Get the sigma for a particular energy"""
        sigmas = u.Quantity([sigma_1TeV, sigma_10TeV])
        energy_nodes = [1, 10] * u.TeV

        log_s = np.log(sigmas.to("deg").value)
        log_en = np.log(energy_nodes.to("TeV").value)
        log_e = np.log(energy.to("TeV").value)
        return np.exp(np.interp(log_e, log_en, log_s)) * u.deg

    def evaluate(
        self, lon, lat, energy, lon_0, lat_0, sigma_1TeV, sigma_10TeV
    ):
        """Evaluate custom Gaussian model"""

        sigma = self.get_sigma(energy, sigma_1TeV, sigma_10TeV)
        sep = angular_separation(lon, lat, lon_0, lat_0)

        exponent = -0.5 * (sep / sigma) ** 2
        norm = 1 / (2 * np.pi * sigma ** 2)
        return norm * np.exp(exponent)

    @property
    def evaluation_radius(self):
        """Evaluation radius (`~astropy.coordinates.Angle`)."""
        return (
            5 * np.max([self.sigma_1TeV.value, self.sigma_10TeV.value]) * u.deg
        )


# Serialisation of this model can be achieved as explained in the previous section.
# You can now use it as stadard `SpatialModel` in your analysis. Note that this is still a `SpatialModel`, and not a `SkyModel`, so it needs to be multiplied by a `SpectralModel` as before. 

# In[ ]:


spatial_model = MyCustomGaussianModel()
spectral_model = PowerLawSpectralModel()
sky_model = SkyModel(
    spatial_model=spatial_model, spectral_model=spectral_model
)


# To visualise it, we evaluate it on a 3D geom. 

# In[ ]:


energy_axis = MapAxis.from_energy_bounds(
    energy_min=0.1 * u.TeV, energy_max=10.0 * u.TeV, nbin=3, name="energy_true"
)
geom = WcsGeom.create(
    skydir=(0, 0), width=5.0 * u.deg, binsz=0.1, axes=[energy_axis]
)
spatial_model.plot_grid(geom=geom, add_cbar=True);

