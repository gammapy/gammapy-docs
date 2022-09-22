
# coding: utf-8

# # Fitting 2D images with Sherpa
# 
# ### Introduction
# 
# Sherpa is the X-ray satellite Chandra modeling and fitting application. It enables the user to construct complex models from simple definitions and fit those models to data, using a variety of statistics and optimization methods. 
# The issues of constraining the source position and morphology are common in X- and Gamma-ray astronomy. 
# This notebook will show you how to apply Sherpa to CTA data.
# 
# Here we will set up Sherpa to fit the counts map and loading the ancillary images for subsequent use. A relevant test statistic for data with Poisson fluctuations is the one proposed by Cash (1979). The simplex (or Nelder-Mead) fitting algorithm is a good compromise between efficiency and robustness. The source fit is best performed in pixel coordinates.
# 
# This tutorial has 2 important parts
# 1. Generating the Maps
# 2. The actual fitting with sherpa.
# 
# Since sherpa deals only with 2-dim images, the first part of this tutorial shows how to prepare gammapy maps to make classical images.

# ### Necessary imports

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import DataStore
from gammapy.irf import make_mean_psf
from gammapy.maps import WcsGeom, MapAxis, Map
from gammapy.cube import MapMaker, PSFKernel


# ### Generate the required Maps
# 
# We first generate the required maps using 3 simulated runs on the Galactic center, exactly as in the [analysis_3d](analysis_3d.ipynb) tutorial.
# 
# It is always advisable to make the maps on fine energy bins, and then sum them over to get an image.

# In[ ]:


# Define which data to use
data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
obs_ids = [110380, 111140, 111159]
observations = data_store.get_observations(obs_ids)


# In[ ]:


energy_axis = MapAxis.from_edges(
    np.logspace(-1, 1.0, 10), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(0, 0),
    binsz=0.02,
    width=(10, 8),
    coordsys="GAL",
    proj="CAR",
    axes=[energy_axis],
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'maker = MapMaker(geom, offset_max=4.0 * u.deg)\nmaps = maker.run(observations)')


# ### Making a PSF Map
# 
# Make a PSF map and weigh it with the exposure at the source position to get a 2D PSF 

# In[ ]:


# mean PSF
src_pos = SkyCoord(0, 0, unit="deg", frame="galactic")
table_psf = make_mean_psf(observations, src_pos)

# PSF kernel used for the model convolution
psf_kernel = PSFKernel.from_table_psf(table_psf, geom, max_radius="0.3 deg")

# get the exposure at the source position
exposure_at_pos = maps["exposure"].get_by_coord(
    {
        "lon": src_pos.l.value,
        "lat": src_pos.b.value,
        "energy": energy_axis.center,
    }
)

# now compute the 2D PSF
psf2D = psf_kernel.make_image(exposures=exposure_at_pos)


# ### Make 2D images from 3D ones
# 
# Since sherpa image fitting works only with 2-dim images,
# we convert the generated maps to 2D images using `run_images()` and save them as fits files. The exposure is weighed with the spectrum before averaging (assumed to be a power law by default).
# 

# In[ ]:


maps = maker.run_images()


# In[ ]:


Path("analysis_3d").mkdir(exist_ok=True)

maps["counts"].write("analysis_3d/counts_2D.fits", overwrite=True)
maps["background"].write("analysis_3d/background_2D.fits", overwrite=True)
maps["exposure"].write("analysis_3d/exposure_2D.fits", overwrite=True)
fits.writeto("analysis_3d/psf_2D.fits", psf2D.data, overwrite=True)


# ### Read the maps and store them in a sherpa model
# 
# We now have the prepared files which sherpa can read. 
# This part of the notebook shows how to do image analysis using sherpa

# In[ ]:


import sherpa.astro.ui as sh

sh.set_stat("cash")
sh.set_method("simplex")

sh.load_image("analysis_3d/counts_2D.fits")
sh.set_coord("logical")

sh.load_table_model("expo", "analysis_3d/exposure_2D.fits")
sh.load_table_model("bkg", "analysis_3d/background_2D.fits")
sh.load_psf("psf", "analysis_3d/psf_2D.fits")


# In principle one might first want to fit the background amplitude. However the background estimation method already yields the correct normalization, so we freeze the background amplitude to unity instead of adjusting it. The (smoothed) residuals from this background model are then computed and shown.

# In[ ]:


sh.set_full_model(bkg)
bkg.ampl = 1
sh.freeze(bkg)


# In[ ]:


resid = Map.read("analysis_3d/counts_2D.fits")
resid.data = sh.get_data_image().y - sh.get_model_image().y
resid_smooth = resid.smooth(width=4)
resid_smooth.plot(add_cbar=True);


# ### Find and fit the brightest source
# We then find the position of the maximum in the (smoothed) residuals map, and fit a (symmetrical) Gaussian source with that initial position:

# In[ ]:


yp, xp = np.unravel_index(
    np.nanargmax(resid_smooth.data), resid_smooth.data.shape
)
ampl = resid_smooth.get_by_pix((xp, yp))[0]

# creates g0 as a gauss2d instance
sh.set_full_model(bkg + psf(sh.gauss2d.g0) * expo)
g0.xpos, g0.ypos = xp, yp
sh.freeze(g0.xpos, g0.ypos)  # fix the position in the initial fitting step

# fix exposure amplitude so that typical exposure is of order unity
expo.ampl = 1e-9
sh.freeze(expo)
sh.thaw(g0.fwhm, g0.ampl)  # in case frozen in a previous iteration

g0.fwhm = 10  # give some reasonable initial values
g0.ampl = ampl


# In[ ]:


get_ipython().run_cell_magic('time', '', 'sh.fit()')


# Fit all parameters of this Gaussian component, fix them and re-compute the residuals map.

# In[ ]:


sh.thaw(g0.xpos, g0.ypos)
sh.fit()
sh.freeze(g0)


# In[ ]:


resid.data = sh.get_data_image().y - sh.get_model_image().y
resid_smooth = resid.smooth(width=3)
resid_smooth.plot();


# ### Iteratively find and fit additional sources
# Instantiate additional Gaussian components, and use them to iteratively fit sources, repeating the steps performed above for component g0. (The residuals map is shown after each additional source included in the model.) This takes some time...

# In[ ]:


# initialize components with fixed, zero amplitude
for i in range(1, 10):
    model = sh.create_model_component("gauss2d", "g" + str(i))
    model.ampl = 0
    sh.freeze(model)

gs = [g0, g1, g2]
sh.set_full_model(bkg + psf(g0 + g1 + g2) * expo)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for i in range(1, len(gs)):\n    yp, xp = np.unravel_index(\n        np.nanargmax(resid_smooth.data), resid_smooth.data.shape\n    )\n    ampl = resid_smooth.get_by_pix((xp, yp))[0]\n    gs[i].xpos, gs[i].ypos = xp, yp\n    gs[i].fwhm = 10\n    gs[i].ampl = ampl\n\n    sh.thaw(gs[i].fwhm)\n    sh.thaw(gs[i].ampl)\n    sh.fit()\n\n    sh.thaw(gs[i].xpos)\n    sh.thaw(gs[i].ypos)\n    sh.fit()\n    sh.freeze(gs[i])\n\n    resid.data = sh.get_data_image().y - sh.get_model_image().y\n    resid_smooth = resid.smooth(width=6)')


# In[ ]:


resid_smooth.plot(add_cbar=True);


# ### Generating output table and Test Statistics estimation
# When adding a new source, one needs to check the significance of this new source. A frequently used method is the Test Statistics (TS). This is done by comparing the change of statistics when the source is included compared to the null hypothesis (no source ; in practice here we fix the amplitude to zero).
# 
# $TS = Cstat(source) - Cstat(no source)$
# 
# The criterion for a significant source detection is typically that it should improve the test statistic by at least 25 or 30. We have added only 3 sources to save time, but you should keep doing this till del(stat) is less than the required number.

# In[ ]:


from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table

rows = []
for g in gs:
    ampl = g.ampl.val
    g.ampl = 0
    stati = sh.get_stat_info()[0].statval
    g.ampl = ampl
    statf = sh.get_stat_info()[0].statval
    delstat = stati - statf

    geom = resid.geom
    # sherpa uses 1 based indexing
    coord = geom.pix_to_coord((g.xpos.val - 1, g.ypos.val - 1))
    pix_scale = geom.pixel_scales.mean().deg
    sigma = g.fwhm.val * pix_scale * gaussian_fwhm_to_sigma
    rows.append(
        dict(delstat=delstat, glon=coord[0], glat=coord[1], sigma=sigma)
    )

table = Table(rows=rows, names=rows[0])
for name in table.colnames:
    table[name].format = ".5g"
table


# ### Exercises
# 
# 1. Keep adding sources till there are no more significat ones in the field. How many Gaussians do you need?
# 2. Use other morphologies for the sources (eg: disk, shell) rather than only Gaussian.
# 3. Compare the TS between different models

# ### More about sherpa
# 
# These are good resources to learn more about Sherpa:
# 
# * https://python4astronomers.github.io/fitting/fitting.html
# * https://github.com/DougBurke/sherpa-standalone-notebooks
# 
# You could read over the examples there, and try to apply a similar analysis to this dataset here to practice.
# 
# If you want a deeper understanding of how Sherpa works, then these proceedings are good introductions:
# 
# * http://conference.scipy.org/proceedings/scipy2009/paper_8/full_text.pdf
# * http://conference.scipy.org/proceedings/scipy2011/pdfs/brefsdal.pdf
