
# coding: utf-8

# # 3D analysis
# 
# This tutorial shows how to run a stacked 3D map-based analysis using three example observations of the Galactic center region with CTA.

# ## Setup

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import DataStore
from gammapy.irf import EnergyDispersion, make_mean_psf, make_mean_edisp
from gammapy.maps import WcsGeom, MapAxis, Map, WcsNDMap
from gammapy.cube import MapMaker, PSFKernel, MapDataset
from gammapy.cube.models import SkyModel, SkyDiffuseCube, BackgroundModel
from gammapy.spectrum.models import PowerLaw, ExponentialCutoffPowerLaw
from gammapy.spectrum import FluxPointsEstimator
from gammapy.image.models import SkyPointSource
from gammapy.utils.fitting import Fit


# ## Prepare modeling input data
# 
# ### Prepare input maps
# 
# We first use the `DataStore` object to access the CTA observations and retrieve a list of observations by passing the observations IDs to the `.get_observations()` method:

# In[ ]:


# Define which data to use and print some information
data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
data_store.info()
print(
    "Total observation time (hours): ",
    data_store.obs_table["ONTIME"].sum() / 3600,
)
print("Observation table: ", data_store.obs_table.colnames)
print("HDU table: ", data_store.hdu_table.colnames)


# In[ ]:


# Select some observations from these dataset by hand
obs_ids = [110380, 111140, 111159]
observations = data_store.get_observations(obs_ids)


# Now we define a reference geometry for our analysis, We choose a WCS based gemoetry with a binsize of 0.02 deg and also define an energy axis: 

# In[ ]:


energy_axis = MapAxis.from_edges(
    np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(0, 0),
    binsz=0.02,
    width=(10, 8),
    coordsys="GAL",
    proj="CAR",
    axes=[energy_axis],
)


# The `MapMaker` object is initialized with this reference geometry and a field of view cut of 4 deg:

# In[ ]:


get_ipython().run_cell_magic('time', '', 'maker = MapMaker(geom, offset_max=4.0 * u.deg)\nmaps = maker.run(observations)')


# The maps are prepared by calling the `.run()` method and passing the `observations`. The `.run()` method returns a Python `dict` containing a `counts`, `background` and `exposure` map:

# In[ ]:


print(maps)


# This is what the summed counts image looks like:

# In[ ]:


counts = maps["counts"].sum_over_axes()
counts.smooth(width=0.1 * u.deg).plot(stretch="sqrt", add_cbar=True, vmax=6);


# This is the background image:

# In[ ]:


background = maps["background"].sum_over_axes()
background.smooth(width=0.1 * u.deg).plot(
    stretch="sqrt", add_cbar=True, vmax=6
);


# And this one the exposure image:

# In[ ]:


exposure = maps["exposure"].sum_over_axes()
exposure.smooth(width=0.1 * u.deg).plot(stretch="sqrt", add_cbar=True);


# We can also compute an excess image just with  a few lines of code:

# In[ ]:


excess = counts - background
excess.smooth(5).plot(stretch="sqrt", add_cbar=True);


# For a more realistic excess plot we can also take into account the diffuse galactic emission. For this tutorial we will load a Fermi diffuse model map that represents a small cutout for the Galactic center region:

# In[ ]:


diffuse_gal = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz")


# In[ ]:


print("Diffuse image: ", diffuse_gal.geom)
print("counts: ", maps["counts"].geom)


# We see that the geometry of the images is completely different, so we need to apply our geometric configuration to the diffuse emission file:

# In[ ]:


coord = maps["counts"].geom.get_coord()

data = diffuse_gal.interp_by_coord(
    {
        "skycoord": coord.skycoord,
        "energy": coord["energy"]
        * maps["counts"].geom.get_axis_by_name("energy").unit,
    },
    interp=3,
)
diffuse_galactic = WcsNDMap(maps["counts"].geom, data)
print("Before: \n", diffuse_gal.geom)
print("Now (same as maps): \n", diffuse_galactic.geom)


# In[ ]:


# diffuse_galactic.slice_by_idx({"energy": 0}).plot(add_cbar=True); # this can be used to check image at different energy bins
diffuse = diffuse_galactic.sum_over_axes()
diffuse.smooth(5).plot(stretch="sqrt", add_cbar=True)
print(diffuse)


# We now multiply the exposure for this diffuse emission to subtract the result from the counts along with the background.

# In[ ]:


combination = diffuse * exposure
combination.unit = ""
combination.smooth(5).plot(stretch="sqrt", add_cbar=True);


# We can plot then the excess image subtracting now the effect of the diffuse galactic emission.

# In[ ]:


excess2 = counts - background - combination

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

axs[0].set_title("With diffuse emission subtraction")
axs[1].set_title("Without diffuse emission subtraction")
excess2.smooth(5).plot(
    cmap="coolwarm", vmin=-1, vmax=1, add_cbar=True, ax=axs[0]
)
excess.smooth(5).plot(
    cmap="coolwarm", vmin=-1, vmax=1, add_cbar=True, ax=axs[1]
);


# ### Prepare IRFs
# 
# To estimate the mean PSF across all observations at a given source position `src_pos`, we use `make_mean_psf()`:

# In[ ]:


# mean PSF
src_pos = SkyCoord(0, 0, unit="deg", frame="galactic")
table_psf = make_mean_psf(observations, src_pos)

# PSF kernel used for the model convolution
psf_kernel = PSFKernel.from_table_psf(table_psf, geom, max_radius="0.3 deg")


# To estimate the mean energy dispersion across all observations at a given source position `src_pos`, we use `make_mean_edisp()`:

# In[ ]:


# define energy grid
energy = energy_axis.edges

# mean edisp
edisp = make_mean_edisp(
    observations, position=src_pos, e_true=energy, e_reco=energy
)


# ### Save maps and IRFs to disk
# 
# It is common to run the preparation step independent of the likelihood fit, because often the preparation of maps, PSF and energy dispersion is slow if you have a lot of data. We first create a folder:

# In[ ]:


path = Path("analysis_3d")
path.mkdir(exist_ok=True)


# And then write the maps and IRFs to disk by calling the dedicated `.write()` methods:

# In[ ]:


# write maps
maps["counts"].write(str(path / "counts.fits"), overwrite=True)
maps["background"].write(str(path / "background.fits"), overwrite=True)
maps["exposure"].write(str(path / "exposure.fits"), overwrite=True)

# write IRFs
psf_kernel.write(str(path / "psf.fits"), overwrite=True)
edisp.write(str(path / "edisp.fits"), overwrite=True)


# ## Likelihood fit
# 
# ### Reading maps and IRFs
# As first step we read in the maps and IRFs that we have saved to disk again:

# In[ ]:


# read maps
maps = {
    "counts": Map.read(str(path / "counts.fits")),
    "background": Map.read(str(path / "background.fits")),
    "exposure": Map.read(str(path / "exposure.fits")),
}

# read IRFs
psf_kernel = PSFKernel.read(str(path / "psf.fits"))
edisp = EnergyDispersion.read(str(path / "edisp.fits"))


# ### Fit mask
# 
# To select a certain energy range for the fit we can create a fit mask:

# In[ ]:


coords = maps["counts"].geom.get_coord()
mask = coords["energy"] > 0.3


# ### Model fit
# 
# No we are ready for the actual likelihood fit. We first define the model as a combination of a point source with a powerlaw:

# In[ ]:


spatial_model = SkyPointSource(lon_0="0.01 deg", lat_0="0.01 deg")
spectral_model = PowerLaw(
    index=2.2, amplitude="3e-12 cm-2 s-1 TeV-1", reference="1 TeV"
)
model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)


# Often, it is useful to fit the normalisation (and also the tilt) of the background. To do so, we have to define the background as a model. In this example, we will keep the tilt fixed and the norm free.

# In[ ]:


background_model = BackgroundModel(maps["background"], norm=1.1, tilt=0.0)
background_model.parameters["norm"].frozen = False
background_model.parameters["tilt"].frozen = True


# Now we set up the `MapDataset` object by passing the prepared maps, IRFs as well as the model:

# In[ ]:


dataset = MapDataset(
    model=model,
    counts=maps["counts"],
    exposure=maps["exposure"],
    background_model=background_model,
    mask_fit=mask,
    psf=psf_kernel,
    edisp=edisp,
)


# No we run the model fit:

# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit = Fit(dataset)\nresult = fit.run(optimize_opts={"print_level": 1})')


# In[ ]:


result.parameters.to_table()


# ### Check model fit
# 
# We check the model fit by computing a residual image. For this we first get the number of predicted counts:

# In[ ]:


npred = dataset.npred()


# And compute a residual image:

# In[ ]:


residual = maps["counts"] - npred


# In[ ]:


residual.sum_over_axes().smooth(width=0.05 * u.deg).plot(
    cmap="coolwarm", vmin=-1, vmax=1, add_cbar=True
);


# We can also plot the best fit spectrum. For that need to extract the covariance of the spectral parameters.

# In[ ]:


spec = model.spectral_model

# set covariance on the spectral model
covariance = result.parameters.covariance
spec.parameters.covariance = covariance[2:5, 2:5]

energy_range = [0.3, 10] * u.TeV
spec.plot(energy_range=energy_range, energy_power=2)
spec.plot_error(energy_range=energy_range, energy_power=2)


# Apparently our model should be improved by adding a component for diffuse Galactic emission and at least one second point source.

# ### Add Galactic diffuse emission to model

# We use both models at the same time, our diffuse model (the same from the Fermi file used before) and our model for the central source. This time, in order to make it more realistic, we will consider an exponential cut off power law spectral model for the source. We will fit again the normalisation and tilt of the background.

# In[ ]:


diffuse_model = SkyDiffuseCube.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz"
)

background_diffuse = BackgroundModel.from_skymodel(
    diffuse_model, exposure=maps["exposure"], psf=psf_kernel
)


# In[ ]:


background_irf = BackgroundModel(maps["background"], norm=1.0, tilt=0.0)
background_total = background_irf + background_diffuse


# In[ ]:


spatial_model = SkyPointSource(lon_0="-0.05 deg", lat_0="-0.05 deg")
spectral_model = ExponentialCutoffPowerLaw(
    index=2 * u.Unit(""),
    amplitude=3e-12 * u.Unit("cm-2 s-1 TeV-1"),
    reference=1.0 * u.TeV,
    lambda_=0.1 / u.TeV,
)

model_ecpl = SkyModel(
    spatial_model=spatial_model,
    spectral_model=spectral_model,
    name="gc-source",
)


# In[ ]:


dataset_combined = MapDataset(
    model=model_ecpl,
    counts=maps["counts"],
    exposure=maps["exposure"],
    background_model=background_total,
    psf=psf_kernel,
    edisp=edisp,
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit_combined = Fit(dataset_combined)\nresult_combined = fit_combined.run()')


# As we can see we have now two components in our model, and we can access them separately.

# In[ ]:


# Checking normalization value (the closer to 1 the better)
print(model_ecpl, "\n")
print(background_irf, "\n")
print(background_diffuse, "\n")


# You can see that the normalisation of the background has vastly improved

# We can now plot the residual image considering this improved model.

# In[ ]:


residual2 = maps["counts"] - dataset_combined.npred()


# Just as a comparison, we can plot our previous residual map (left) and the new one (right) with the same scale:

# In[ ]:


plt.figure(figsize=(15, 5))
ax_1 = plt.subplot(121, projection=residual.geom.wcs)
ax_2 = plt.subplot(122, projection=residual.geom.wcs)

ax_1.set_title("Without diffuse emission subtraction")
ax_2.set_title("With diffuse emission subtraction")

residual.sum_over_axes().smooth(width=0.05 * u.deg).plot(
    cmap="coolwarm", vmin=-1, vmax=1, add_cbar=True, ax=ax_1
)
residual2.sum_over_axes().smooth(width=0.05 * u.deg).plot(
    cmap="coolwarm", vmin=-1, vmax=1, add_cbar=True, ax=ax_2
);


# ## Computing Flux Points
# 
# Finally we compute flux points for the galactic center source. For this we first define an energy binning:

# In[ ]:


e_edges = [0.3, 1, 3, 10] * u.TeV
fpe = FluxPointsEstimator(
    datasets=[dataset_combined], e_edges=e_edges, source="gc-source"
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'flux_points = fpe.run()')


# Now let's plot the best fit model and flux points:

# In[ ]:


flux_points.table["is_ul"] = flux_points.table["ts"] < 4
ax = flux_points.plot(energy_power=2)
model_ecpl.spectral_model.plot(
    ax=ax, energy_range=energy_range, energy_power=2
);


# ## Summary
# 
# Note that this notebook aims to show you the procedure of a 3D analysis using just a few observations and a cutted Fermi model. Results get much better for a more complete analysis considering the GPS dataset from the CTA First Data Challenge (DC-1) and also the CTA model for the Galactic diffuse emission, as shown in the next image:

# ![](images/DC1_3d.png)

# The complete tutorial notebook of this analysis is available to be downloaded in [GAMMAPY-EXTRA](https://github.com/gammapy/gammapy-extra) repository at https://github.com/gammapy/gammapy-extra/blob/master/analyses/cta_1dc_gc_3d.ipynb).

# ## Exercises
# 
# * Analyse the second source in the field of view: G0.9+0.1 and add it to the combined model.
