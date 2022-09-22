#!/usr/bin/env python
# coding: utf-8

# # Fermi-LAT data with Gammapy
# 
# ## Introduction
# 
# This tutorial will show you how to work with Fermi-LAT data with Gammapy. As an example, we will look at the Galactic center region using the high-energy dataset that was used for the 3FHL catalog, in the energy range 10 GeV to 2 TeV.
# 
# We note that support for Fermi-LAT data analysis in Gammapy is very limited. For most tasks, we recommend you use 
# [Fermipy](http://fermipy.readthedocs.io/), which is based on the [Fermi Science Tools](https://fermi.gsfc.nasa.gov/ssc/data/analysis/software/) (Fermi ST).
# 
# Using Gammapy with Fermi-LAT data could be an option for you if you want to do an analysis that is not easily possible with Fermipy and the Fermi Science Tools. For example a joint likelihood fit of Fermi-LAT data with data e.g. from H.E.S.S., MAGIC, VERITAS or some other instrument, or analysis of Fermi-LAT data with a complex spatial or spectral model that is not available in Fermipy or the Fermi ST.
# 
# Besides Gammapy, you might want to look at are [Sherpa](http://cxc.harvard.edu/sherpa/) or [3ML](https://threeml.readthedocs.io/). Or just using Python to roll your own analyis using several existing analysis packages. E.g. it it possible to use Fermipy and the Fermi ST to evaluate the likelihood on Fermi-LAT data, and Gammapy to evaluate it e.g. for IACT data, and to do a joint likelihood fit using e.g. [iminuit](http://iminuit.readthedocs.io/) or [emcee](http://dfm.io/emcee).
# 
# To use Fermi-LAT data with Gammapy, you first have to use the Fermi ST to prepare an event list (using ``gtselect`` and ``gtmktime``, exposure cube (using ``gtexpcube2`` and PSF (using ``gtpsf``). You can then use [gammapy.data.EventList](https://docs.gammapy.org/dev/api/gammapy.data.EventList.html), [gammapy.maps](https://docs.gammapy.org/dev/maps/index.html) and the [gammapy.irf.EnergyDependentTablePSF](https://docs.gammapy.org/dev/api/gammapy.irf.EnergyDependentTablePSF.html) to read the Fermi-LAT maps and PSF, i.e. support for these high-level analysis products from the Fermi ST is built in. To do a 3D map analyis, you can use Fit for Fermi-LAT data in the same way that it's use for IACT data. This is illustrated in this notebook. A 1D region-based spectral analysis is also possible, this will be illustrated in a future tutorial.
# 
# ## Setup
# 
# **IMPORTANT**: For this notebook you have to get the prepared ``3fhl`` dataset provided in your $GAMMAPY_DATA.
# 
# Note that the ``3fhl`` dataset is high-energy only, ranging from 10 GeV to 2 TeV.

# In[ ]:


# Check that you have the prepared Fermi-LAT dataset
# We will use diffuse models from here
get_ipython().system('ls -1 $GAMMAPY_DATA/fermi_3fhl')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.data import EventList
from gammapy.irf import EnergyDependentTablePSF, EnergyDispersion
from gammapy.maps import Map, MapAxis, WcsNDMap, WcsGeom
from gammapy.modeling.models import (
    TemplateSpectralModel,
    PowerLawSpectralModel,
    PointSpatialModel,
    ConstantSpatialModel,
    SkyModel,
    SkyDiffuseCube,
    SkyModels,
)
from gammapy.cube import MapDataset, PSFKernel, MapEvaluator
from gammapy.modeling import Fit


# ## Events
# 
# To load up the Fermi-LAT event list, use the [gammapy.data.EventList](https://docs.gammapy.org/dev/api/gammapy.data.EventList.html) class:

# In[ ]:


events = EventList.read(
    "$GAMMAPY_DATA/fermi_3fhl/fermi_3fhl_events_selected.fits.gz"
)
print(events)


# The event data is stored in a [astropy.table.Table](http://docs.astropy.org/en/stable/api/astropy.table.Table.html) object. In case of the Fermi-LAT event list this contains all the additional information on positon, zenith angle, earth azimuth angle, event class, event type etc.

# In[ ]:


events.table.colnames


# In[ ]:


events.table[:5][["ENERGY", "RA", "DEC"]]


# In[ ]:


print(events.time[0].iso)
print(events.time[-1].iso)


# In[ ]:


energy = events.energy
energy.info("stats")


# As a short analysis example we will count the number of events above a certain minimum energy: 

# In[ ]:


for e_min in [10, 100, 1000] * u.GeV:
    n = (events.energy > e_min).sum()
    print("Events above {0:4.0f}: {1:5.0f}".format(e_min, n))


# ## Counts
# 
# Let us start to prepare things for an 3D map analysis of the Galactic center region with Gammapy. The first thing we do is to define the map geometry. We chose a TAN projection centered on position ``(glon, glat) = (0, 0)`` with pixel size 0.1 deg, and four energy bins.

# In[ ]:


gc_pos = SkyCoord(0, 0, unit="deg", frame="galactic")
energy_axis = MapAxis.from_edges(
    [10, 30, 100, 300, 2000], name="energy", unit="GeV", interp="log"
)
counts = Map.create(
    skydir=gc_pos,
    npix=(100, 80),
    proj="TAN",
    coordsys="GAL",
    binsz=0.1,
    axes=[energy_axis],
    dtype=float,
)
# We put this call into the same Jupyter cell as the Map.create
# because otherwise we could accidentally fill the counts
# multiple times when executing the ``fill_by_coord`` multiple times.
counts.fill_by_coord({"skycoord": events.radec, "energy": events.energy})


# In[ ]:


counts.geom.axes[0]


# In[ ]:


counts.sum_over_axes().smooth(2).plot(stretch="sqrt", vmax=30);


# ## Exposure
# 
# The Fermi-LAT datatset contains the energy-dependent exposure for the whole sky as a HEALPix map computed with ``gtexpcube2``. This format is supported by ``gammapy.maps`` directly.
# 
# Interpolating the exposure cube from the Fermi ST to get an exposure cube matching the spatial geometry and energy axis defined above with Gammapy is easy. The only point to watch out for is how exactly you want the energy axis and binning handled.
# 
# Below we just use the default behaviour, which is linear interpolation in energy on the original exposure cube. Probably log interpolation would be better, but it doesn't matter much here, because the energy binning is fine. Finally, we just copy the counts map geometry, which contains an energy axis with `node_type="edges"`. This is non-ideal for exposure cubes, but again, acceptable because exposure doesn't vary much from bin to bin, so the exact way interpolation occurs in later use of that exposure cube doesn't matter a lot. Of course you could define any energy axis for your exposure cube that you like.

# In[ ]:


exposure_hpx = Map.read(
    "$GAMMAPY_DATA/fermi_3fhl/fermi_3fhl_exposure_cube_hpx.fits.gz"
)
# Unit is not stored in the file, set it manually
exposure_hpx.unit = "cm2 s"
print(exposure_hpx.geom)
print(exposure_hpx.geom.axes[0])


# In[ ]:


exposure_hpx.plot();


# In[ ]:


# For exposure, we choose a geometry with node_type='center',
# whereas for counts it was node_type='edge'
axis = MapAxis.from_nodes(
    counts.geom.axes[0].center, name="energy", unit="GeV", interp="log"
)
geom = WcsGeom(wcs=counts.geom.wcs, npix=counts.geom.npix, axes=[axis])

coord = counts.geom.get_coord()
data = exposure_hpx.interp_by_coord(coord)


# In[ ]:


exposure = WcsNDMap(geom, data, unit=exposure_hpx.unit, dtype=float)
print(exposure.geom)
print(exposure.geom.axes[0])


# In[ ]:


# Exposure is almost constant accross the field of view
exposure.slice_by_idx({"energy": 0}).plot(add_cbar=True);


# In[ ]:


# Exposure varies very little with energy at these high energies
energy = [10, 100, 1000] * u.GeV
exposure.get_by_coord({"skycoord": gc_pos, "energy": energy})


# ## Galactic diffuse background

# The Fermi-LAT collaboration provides a galactic diffuse emission model, that can be used as a background model for
# Fermi-LAT source analysis.
# 
# Diffuse model maps are very large (100s of MB), so as an example here, we just load one that represents a small cutout for the Galactic center region.

# In[ ]:


diffuse_galactic_fermi = Map.read(
    "$GAMMAPY_DATA/fermi_3fhl/gll_iem_v06_cutout.fits"
)
# Unit is not stored in the file, set it manually
diffuse_galactic_fermi.unit = "cm-2 s-1 MeV-1 sr-1"
print(diffuse_galactic_fermi.geom)
print(diffuse_galactic_fermi.geom.axes[0])


# In[ ]:


# Interpolate the diffuse emission model onto the counts geometry
# The resolution of `diffuse_galactic_fermi` is low: bin size = 0.5 deg
# We use ``interp=3`` which means cubic spline interpolation
coord = counts.geom.get_coord()

data = diffuse_galactic_fermi.interp_by_coord(
    {"skycoord": coord.skycoord, "energy": coord["energy"]}, interp=3
)
diffuse_galactic = WcsNDMap(
    exposure.geom, data, unit=diffuse_galactic_fermi.unit
)

print(diffuse_galactic.geom)
print(diffuse_galactic.geom.axes[0])


# In[ ]:


diffuse_galactic.slice_by_idx({"energy": 0}).plot(add_cbar=True);


# In[ ]:


# Exposure varies very little with energy at these high energies
energy = np.logspace(1, 3, 10) * u.GeV
dnde = diffuse_galactic.interp_by_coord(
    {"skycoord": gc_pos, "energy": energy}, interp="linear", fill_value=None
)
plt.plot(energy.value, dnde, "*")
plt.loglog()
plt.xlabel("Energy (GeV)")
plt.ylabel("Flux (cm-2 s-1 MeV-1 sr-1)")


# In[ ]:


# TODO: show how one can fix the extrapolate to high energy
# by computing and padding an extra plane e.g. at 1e3 TeV
# that corresponds to a linear extrapolation


# ## Isotropic diffuse background
# 
# To load the isotropic diffuse model with Gammapy, use the [gammapy.modeling.models.TemplateSpectralModel](https://docs.gammapy.org/dev/api/gammapy.modeling.models.TemplateSpectralModel.html). We are using `'fill_value': 'extrapolate'` to extrapolate the model above 500 GeV:

# In[ ]:


diffuse_iso = TemplateSpectralModel.read_fermi_isotropic_model(
    filename="$GAMMAPY_DATA/fermi_3fhl/iso_P8R2_SOURCE_V6_v06.txt",
    interp_kwargs={"fill_value": None},
)


# We can plot the model in the energy range between 50 GeV and 2000 GeV:

# In[ ]:


erange = [50, 2000] * u.GeV
diffuse_iso.plot(erange, flux_unit="1 / (cm2 MeV s sr)");


# ## PSF
# 
# Next we will tke a look at the PSF. It was computed using ``gtpsf``, in this case for the Galactic center position. Note that generally for Fermi-LAT, the PSF only varies little within a given regions of the sky, especially at high energies like what we have here. We use the [gammapy.irf.EnergyDependentTablePSF](https://docs.gammapy.org/dev/api/gammapy.irf.EnergyDependentTablePSF.html) class to load the PSF and use some of it's methods to get some information about it.

# In[ ]:


psf = EnergyDependentTablePSF.read(
    "$GAMMAPY_DATA/fermi_3fhl/fermi_3fhl_psf_gc.fits.gz"
)
print(psf)


# To get an idea of the size of the PSF we check how the containment radii of the Fermi-LAT PSF vari with energy and different containment fractions:

# In[ ]:


plt.figure(figsize=(8, 5))
psf.plot_containment_vs_energy(linewidth=2, fractions=[0.68, 0.95])
plt.xlim(50, 2000)
plt.show()


# In addition we can check how the actual shape of the PSF varies with energy and compare it against the mean PSF between 50 GeV and 2000 GeV:

# In[ ]:


plt.figure(figsize=(8, 5))

for energy in [100, 300, 1000] * u.GeV:
    psf_at_energy = psf.table_psf_at_energy(energy)
    psf_at_energy.plot_psf_vs_rad(label="PSF @ {:.0f}".format(energy), lw=2)

erange = [50, 2000] * u.GeV
spectrum = PowerLawSpectralModel(index=2.3)
psf_mean = psf.table_psf_in_energy_band(energy_band=erange, spectrum=spectrum)
psf_mean.plot_psf_vs_rad(label="PSF Mean", lw=4, c="k", ls="--")

plt.xlim(1e-3, 0.3)
plt.ylim(1e3, 1e6)
plt.legend();


# In[ ]:


# Let's compute a PSF kernel matching the pixel size of our map
psf_kernel = PSFKernel.from_table_psf(psf, counts.geom, max_radius="1 deg")


# In[ ]:


psf_kernel.psf_kernel_map.sum_over_axes().plot(stretch="log", add_cbar=True);


# ### Energy Dispersion
# For simplicity we assume a diagonal energy dispersion:

# In[ ]:


e_true = exposure.geom.axes[0].edges
e_reco = counts.geom.axes[0].edges
edisp = EnergyDispersion.from_diagonal_response(e_true=e_true, e_reco=e_reco)


# ## Background
# 
# Let's compute a background cube, with predicted number of background events per pixel from the diffuse Galactic and isotropic model components. For this, we use the use the [gammapy.cube.MapEvaluator](https://docs.gammapy.org/dev/api/gammapy.cube.MapEvaluator.html) to multiply with the exposure and apply the PSF. The Fermi-LAT energy dispersion at high energies is small, we neglect it here.

# In[ ]:


model_diffuse = SkyDiffuseCube(diffuse_galactic, name="diffuse")
eval_diffuse = MapEvaluator(
    model=model_diffuse, exposure=exposure, psf=psf_kernel, edisp=edisp
)

background_gal = eval_diffuse.compute_npred()

background_gal.sum_over_axes().plot()
print("Background counts from Galactic diffuse: ", background_gal.data.sum())


# In[ ]:


model_iso = SkyModel(ConstantSpatialModel(), diffuse_iso, name="diffuse-iso")

eval_iso = MapEvaluator(model=model_iso, exposure=exposure, edisp=edisp)

background_iso = eval_iso.compute_npred()

background_iso.sum_over_axes().plot(add_cbar=True)
print("Background counts from isotropic diffuse: ", background_iso.data.sum())


# In[ ]:


background_total = background_iso + background_gal


# ## Excess and flux
# 
# Let's compute an excess and flux image, by subtracting the background, and summing over the energy axis.

# In[ ]:


excess = counts.copy()
excess.data -= background_total.data
excess.sum_over_axes().smooth("0.1 deg").plot(
    cmap="coolwarm", vmin=-5, vmax=5, add_cbar=True
)
print("Excess counts: ", excess.data.sum())


# In[ ]:


flux = excess.copy()
flux.data /= exposure.data
flux.unit = excess.unit / exposure.unit
flux.sum_over_axes().smooth("0.1 deg").plot(stretch="sqrt", add_cbar=True);


# ## Fit
# 
# Finally, the big finale: let's do a 3D map fit for the source at the Galactic center, to measure it's position and spectrum. We keep the background normalization free.

# In[ ]:


model = SkyModel(
    PointSpatialModel("0 deg", "0 deg", frame="galactic"),
    PowerLawSpectralModel(
        index=2.5, amplitude="1e-11 cm-2 s-1 TeV-1", reference="100 GeV"
    ),
)

model_total = SkyModels([model, model_diffuse, model_iso])

dataset = MapDataset(
    model=model_total, counts=counts, exposure=exposure, psf=psf_kernel
)
fit = Fit(dataset)
result = fit.run()


# In[ ]:


print(result)


# In[ ]:


dataset.parameters.to_table()


# In[ ]:


residual = counts - dataset.npred()
residual.sum_over_axes().smooth("0.1 deg").plot(
    cmap="coolwarm", vmin=-3, vmax=3, add_cbar=True
);


# ## Exercises
# 
# - Fit the position and spectrum of the source [SNR G0.9+0.1](http://gamma-sky.net/#/cat/tev/110).
# - Make maps and fit the position and spectrum of the [Crab nebula](http://gamma-sky.net/#/cat/tev/25).

# ## Summary
# 
# In this tutorial you have seen how to work with Fermi-LAT data with Gammapy. You have to use the Fermi ST to prepare the exposure cube and PSF, and then you can use Gammapy for any event or map analysis using the same methods that are used to analyse IACT data.
# 
# This works very well at high energies (here above 10 GeV), where the exposure and PSF is almost constant spatially and only varies a little with energy. It is not expected to give good results for low-energy data, where the Fermi-LAT PSF is very large. If you are interested to help us validate down to what energy Fermi-LAT analysis with Gammapy works well (e.g. by re-computing results from 3FHL or other published analysis results), or to extend the Gammapy capabilities (e.g. to work with energy-dependent multi-resolution maps and PSF), that would be very welcome!
