
# coding: utf-8

# # Cube analysis with Gammapy (part 2)
# 
# ## Introduction 
# 
# In this tutorial we will learn how to compute a morphological and spectral fit simultanously.
# 
# This is part 2 of 2 for IACT cube analysis. If you haven't prepared your cubes, PSF and EDISP yet, do part 1 first.
# 
# The fitting is done using [Sherpa](http://cxc.harvard.edu/contrib/sherpa47/).
# 
# We will use the following classes:
# 
# - [gammapy.cube](http://docs.gammapy.org/en/latest/cube/index.html) where are strored the counts, the background, the exposure and the mean psf of this Crab dataset.
# - [gammapy.irf.EnergyDispersion](http://docs.gammapy.org/en/latest/api/gammapy.irf.EnergyDispersion.html) where is stored the mean rmf of this Crab dataset.
# -  the method [gammapy.cube.SkyCube.to_sherpa_data3d](http://docs.gammapy.org/en/latest/api/gammapy.cube.SkyCube.html#gammapy.cube.SkyCube.to_sherpa_data3d) to transform the counts cube in Sherpa object.
# - [gammapy.cube.CombinedModel3DInt](http://docs.gammapy.org/en/latest/api/gammapy.cube.CombinedModel3DInt.html) to combine the spectral and spatial model for the fit if you consider a perfect energy resolution
# - [gammapy.cube.CombinedModel3DIntConvolveEdisp](http://docs.gammapy.org/en/latest/api/gammapy.cube.CombinedModel3DIntConvolveEdisp.html) to combine the spectral and spatial model for the fit taking into account that the true energy is different than the reconstructed one.
# 
# We will use the cubes built on the 4 Crab observations of gammapy-extra.
# 
# You could use the cubes we just created with the notebook `cube_analysis_part1.ipynb`

# ## Setup
# 
# As always, we start with some notebook setup and imports

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import logging
logging.getLogger("sherpa.fit").setLevel(logging.ERROR)


# In[2]:


import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion

from gammapy.extern.pathlib import Path
from gammapy.irf import EnergyDispersion
from gammapy.cube import SkyCube
from gammapy.cube.sherpa_ import (
    CombinedModel3DInt,
    CombinedModel3DIntConvolveEdisp,
    NormGauss2DInt,
)

from sherpa.models import PowLaw1D, TableModel
from sherpa.estmethods import Covariance
from sherpa.optmethods import NelderMead
from sherpa.stats import Cash
from sherpa.fit import Fit


# ## 3D analysis assuming that there is no energy dispersion (perfect energy resolution)
# 
# ### Load the different cubes needed for the analysis
# 
# We will use the Cubes build on the 4 Crab observations of gammapy-extra. You could use the Cubes we just created with the notebook cube_analysis.ipynb by changing the cube_directory by your local path.
# 
# - Counts cube

# In[3]:


cube_dir = Path('$GAMMAPY_EXTRA/test_datasets/cube')
counts_3d = SkyCube.read(cube_dir / 'counts_cube.fits')
# Transformation to a sherpa object
cube = counts_3d.to_sherpa_data3d(dstype='Data3DInt')


# - Background Cube

# In[4]:


bkg_3d = SkyCube.read(cube_dir / 'bkg_cube.fits')
bkg = TableModel('bkg')
bkg.load(None, bkg_3d.data.value.ravel())
bkg.ampl = 1
bkg.ampl.freeze()


# - Exposure Cube

# In[5]:


exposure_3d = SkyCube.read(cube_dir / 'exposure_cube.fits')
i_nan = np.where(np.isnan(exposure_3d.data))
exposure_3d.data[i_nan] = 0
# In order to have the exposure in cm2 s
exposure_3d.data = exposure_3d.data * u.Unit('m2 / cm2').to('')


# - PSF Cube

# In[6]:


psf_3d = SkyCube.read(cube_dir / 'psf_cube.fits')


# ### Setup Combined spatial and spectral model

# In[7]:


# Define a 2D gaussian for the spatial model
spatial_model = NormGauss2DInt('spatial-model')

# Define a power law for the spectral model
spectral_model = PowLaw1D('spectral-model')

# Combine spectral and spatial model
coord = counts_3d.sky_image_ref.coordinates(mode="edges")
energies = counts_3d.energies(mode='edges').to("TeV")
# Here the source model will be convolve by the psf:
# PSF * (source_model * exposure)
source_model = CombinedModel3DInt(
    coord=coord,
    energies=energies,
    use_psf=True,
    exposure=exposure_3d,
    psf=psf_3d,
    spatial_model=spatial_model,
    spectral_model=spectral_model,
)


# ### Set starting value

# In[8]:


center = SkyCoord(83.633083, 22.0145, unit="deg").galactic
source_model.gamma = 2.2
source_model.xpos = center.l.value
source_model.ypos = center.b.value
source_model.fwhm = 0.12
source_model.ampl = 1.0


# ### Fit

# In[9]:


# Define the model
flux_factor = 1e-11
model = bkg + flux_factor * source_model


# In[10]:


# Fit to the counts Cube sherpa object
fit = Fit(
    data=cube,
    model=model,
    stat=Cash(),
    method=NelderMead(),
    estmethod=Covariance(),
)
fit_results = fit.fit()
print(fit_results.format())


# In[11]:


err_est_results = fit.est_errors()
print(err_est_results.format())


# ## Add an exlusion mask for the Fit
# For example if you want to exclude a region in the FoV of your cube, you just have to provide a SkyCube with the same dimension than the Counts cube and with 0 in the region you want to exlude and 1 outside. With this SkyCube mask you can select only some energy band for the fit or just some spatial region whatever the energy band or both.

# ### Define the mask
# Here this is a test case, there is no source to exlude in our FOV but we will create a mask that remove some events from the source. You just see at the end that the amplitude fitted in the 3D analysis is lower than the one when you use all the events from the Crab. The principle works even if this is not usefull here, just a testcase.

# In[12]:


exclusion_region = CircleSkyRegion(
    center=SkyCoord(83.60, 21.88, unit='deg'),
    radius=Angle(0.1, "deg"),
)

sky_mask_cube = counts_3d.region_mask(exclusion_region)
sky_mask_cube.data = sky_mask_cube.data.astype(int)
index_region_selected_3d = np.where(sky_mask_cube.data.value == 1)


# Add the mask to the data

# In[13]:


# Set the counts
cube = counts_3d.to_sherpa_data3d(dstype='Data3DInt')
cube.mask = sky_mask_cube.data.value.ravel()


# Select only the background pixels of the cube of the selected region to create the background model

# In[14]:


# Set the bkg and select only the data points of the selected region
bkg = TableModel('bkg')
bkg.load(None, bkg_3d.data.value[index_region_selected_3d].ravel())
bkg.ampl = 1
bkg.ampl.freeze()


# Add the indices of the selected region of the Cube to combine the model

# In[15]:


# The model is evaluated on all the points then it is compared with the data only on the selected_region
source_model = CombinedModel3DInt(
    coord=coord,
    energies=energies,
    use_psf=True,
    exposure=exposure_3d,
    psf=psf_3d,
    spatial_model=spatial_model,
    spectral_model=spectral_model,
    select_region=True,
    index_selected_region=index_region_selected_3d,
)

# Set starting values
source_model.gamma = 2.2
source_model.xpos = center.l.value
source_model.ypos = center.b.value
source_model.fwhm = 0.12
source_model.ampl = 1.0

# Define the model
model = bkg + flux_factor * (source_model)


# In[16]:


fit = Fit(
    data=cube,
    model=model,
    stat=Cash(),
    method=NelderMead(),
    estmethod=Covariance(),
)
fit_results = fit.fit()
print(fit_results.format())


# In[17]:


err_est_results = fit.est_errors()

print(err_est_results.format())


# The fitted flux is less than in the previous example since here the mask remove some events from the Crab

# ## 3D analysis taking into account the energy dispersion

# In[18]:


# Set the counts
counts_3d = SkyCube.read(cube_dir / "counts_cube.fits")
cube = counts_3d.to_sherpa_data3d(dstype='Data3DInt')

# Set the bkg
bkg_3d = SkyCube.read(cube_dir / 'bkg_cube.fits')
bkg = TableModel('bkg')
bkg.load(None, bkg_3d.data.value.ravel())
bkg.ampl = 1
bkg.ampl.freeze()

# Set the exposure
exposure_3d = SkyCube.read(cube_dir / 'exposure_cube_etrue.fits')
i_nan = np.where(np.isnan(exposure_3d.data))
exposure_3d.data[i_nan] = 0
exposure_3d.data = exposure_3d.data * 1e4

# Set the mean psf model
psf_3d = SkyCube.read(cube_dir / 'psf_cube_etrue.fits')

# Load the mean rmf calculated for the 4 Crab runs
rmf = EnergyDispersion.read(cube_dir / 'rmf.fits')


# In[19]:


# Setup combined spatial and spectral model
spatial_model = NormGauss2DInt('spatial-model')
spectral_model = PowLaw1D('spectral-model')


# ### Add the mean RMF to the Combine3DInt object 
# 
# The model is evaluated on the true energy bin then it is convolved by the energy dispersion to compare to the counts data in reconstructed energy

# In[20]:


coord = counts_3d.sky_image_ref.coordinates(mode="edges")
energies = counts_3d.energies(mode='edges').to("TeV")
source_model = CombinedModel3DIntConvolveEdisp(
    coord=coord,
    energies=energies,
    use_psf=True,
    exposure=exposure_3d,
    psf=psf_3d,
    spatial_model=spatial_model,
    spectral_model=spectral_model,
    edisp=rmf.data.data,
)

# Set starting values
center = SkyCoord(83.633083, 22.0145, unit="deg").galactic
source_model.gamma = 2.2
source_model.xpos = center.l.value
source_model.ypos = center.b.value
source_model.fwhm = 0.12
source_model.ampl = 1.0

# Define the model
model = bkg + flux_factor * source_model


# In[21]:


fit = Fit(
    data=cube,
    model=model,
    stat=Cash(),
    method=NelderMead(),
    estmethod=Covariance(),
)
fit_results = fit.fit()
print(fit_results.format())


# In[22]:


err_est_results = fit.est_errors()
print(err_est_results.format())


# ## Discussion
# 
# Here the Cubes are constructed from dummy data. We don't expect to find the good spectral shape or amplitude since there is a lot of problem with the PFS, rmf etc.. in these dummy data.
# On real data, we find the good Crab position and spectral shape for a power law or a power law with an exponential cutoff....

# In[23]:


# TODO: there should be some visualisation of results.
# E.g. a plotted spectral model + butterfly
# Or a counts, model and residual image

