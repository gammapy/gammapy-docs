#!/usr/bin/env python
# coding: utf-8

# # 3D simulation and fitting
# 
# This tutorial shows how to do a 3D map-based simulation and fit.
# 
# For a tutorial on how to do a 3D map analyse of existing data, see the `analysis_3d` tutorial.
# 
# This can be useful to do a performance / sensitivity study, or to evaluate the capabilities of Gammapy or a given analysis method. Note that is is a binned simulation as is e.g. done also in Sherpa for Chandra, not an event sampling and anbinned analysis as is done e.g. in the Fermi ST or ctools.

# ## Imports and versions

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from gammapy.irf import load_cta_irfs
from gammapy.maps import WcsGeom, MapAxis, WcsNDMap
from gammapy.spectrum.models import PowerLaw
from gammapy.image.models import SkyGaussian
from gammapy.cube.models import SkyModel, BackgroundModel
from gammapy.cube import MapDataset, PSFKernel
from gammapy.cube import make_map_exposure_true_energy, make_map_background_irf
from gammapy.utils.fitting import Fit
from gammapy.data import FixedPointingInfo


# In[ ]:


get_ipython().system('gammapy info --no-envvar --no-dependencies --no-system')


# ## Simulate

# In[ ]:


filename = (
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)
irfs = load_cta_irfs(filename)


# In[ ]:


# Define sky model to simulate the data
spatial_model = SkyGaussian(lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.3 deg")
spectral_model = PowerLaw(
    index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)
sky_model = SkyModel(
    spatial_model=spatial_model, spectral_model=spectral_model
)
print(sky_model)


# In[ ]:


# Define map geometry
axis = MapAxis.from_edges(
    np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(0, 0), binsz=0.02, width=(5, 4), coordsys="GAL", axes=[axis]
)


# In[ ]:


# Define some observation parameters
# We read in the pointing info from one of the 1dc event list files as an example
pointing = FixedPointingInfo.read(
    "$GAMMAPY_DATA/cta-1dc/data/baseline/gps/gps_baseline_110380.fits"
)

livetime = 1 * u.hour
offset_max = 2 * u.deg
offset = Angle("2 deg")


# In[ ]:


exposure = make_map_exposure_true_energy(
    pointing=pointing.radec, livetime=livetime, aeff=irfs["aeff"], geom=geom
)
exposure.slice_by_idx({"energy": 3}).plot(add_cbar=True);


# In[ ]:


background = make_map_background_irf(
    pointing=pointing, ontime=livetime, bkg=irfs["bkg"], geom=geom
)
background.slice_by_idx({"energy": 3}).plot(add_cbar=True);


# In[ ]:


psf = irfs["psf"].to_energy_dependent_table_psf(theta=offset)
psf_kernel = PSFKernel.from_table_psf(psf, geom, max_radius=0.3 * u.deg)
psf_kernel.psf_kernel_map.sum_over_axes().plot(stretch="log");


# In[ ]:


energy = axis.edges
edisp = irfs["edisp"].to_energy_dispersion(
    offset, e_reco=energy, e_true=energy
)
edisp.plot_matrix();


# Now we have to compute `npred`  maps, i.e. "predicted counts per pixel" given the model and the observation infos: exposure, background, PSF and EDISP. For this we use the `MapDataset` object:

# In[ ]:


background_model = BackgroundModel(background)
dataset = MapDataset(
    model=sky_model,
    exposure=exposure,
    background_model=background_model,
    psf=psf_kernel,
    edisp=edisp,
)


# In[ ]:


npred = dataset.npred()


# In[ ]:


npred.sum_over_axes().plot(add_cbar=True);


# In[ ]:


# This one line is the core of how to simulate data when
# using binned simulation / analysis: you Poisson fluctuate
# npred to obtain simulated observed counts.
# Compute counts as a Poisson fluctuation
rng = np.random.RandomState(seed=42)
counts = rng.poisson(npred.data)
counts_map = WcsNDMap(geom, counts)


# In[ ]:


counts_map.sum_over_axes().plot();


# ## Fit
# 
# Now let's analyse the simulated data.
# Here we just fit it again with the same model we had before, but you could do any analysis you like here, e.g. fit a different model, or do a region-based analysis, ...

# In[ ]:


# Define sky model to fit the data
spatial_model = SkyGaussian(lon_0="0.1 deg", lat_0="0.1 deg", sigma="0.5 deg")
spectral_model = PowerLaw(
    index=2, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)
model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)
print(model)


# In[ ]:


# We do not want to fit the background in this case, so we will freeze the parameters
background_model.parameters["norm"].value = 1.0
background_model.parameters["norm"].frozen = True
background_model.parameters["tilt"].frozen = True

print(background_model)


# In[ ]:


dataset = MapDataset(
    model=model,
    exposure=exposure,
    counts=counts_map,
    background_model=background_model,
    psf=psf_kernel,
    edisp=edisp,
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit = Fit(dataset)\nresult = fit.run(optimize_opts={"print_level": 1})')


# True model:

# In[ ]:


print(sky_model)


# Best-fit model:

# In[ ]:


print(model)


# To get the errors on the model, we can check the covariance table:

# In[ ]:


result.parameters.covariance_to_table()


# In[ ]:


# Or, to see the value of and error on an individual parameter, say index:
print(result.parameters["index"].value, result.parameters.error("index"))


# In[ ]:


# TODO: show e.g. how to make a residual image

