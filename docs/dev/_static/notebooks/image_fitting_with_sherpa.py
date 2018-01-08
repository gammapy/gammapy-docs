
# coding: utf-8

# # CTA 2D source fitting with Sherpa
# F. Acero & Y. Gallant
# October 2017

# ## Introduction
# 
# Sherpa is the X-ray satellite Chandra modeling and fitting application. It enables the user to construct complex models from simple definitions and fit those models to data, using a variety of statistics and optimization methods. 
# The issues of constraining the source position and morphology are common in X- and Gamma-ray astronomy. 
# This notebook will show you how to apply Sherpa to CTA data.
# 
# Here we will set up Sherpa to fit the counts map and loading the ancillary images for subsequent use. A relevant test statistic for data with Poisson fluctuations is the one proposed by Cash (1979). The simplex (or Nelder-Mead) fitting algorithm is a good compromise between efficiency and robustness. The source fit is best performed in pixel coordinates.

# ## Read sky images
# The sky image that are loaded here have been prepared in a separated notebook. Here we start from those fits file and focus on the source fitting aspect.
# 
# The info needed for sherpa are:
# - Count map
# - Background map
# - Exposure map
# - PSF map
# 
# For info, the fits file are written in the following way in the Sky map generation notebook:
# 
# ```
# images['counts']    .write("G300-0_test_counts.fits", clobber=True)
# images['exposure']  .write("G300-0_test_exposure.fits", clobber=True)
# images['background'].write("G300-0_test_background.fits", clobber=True)
# 
# ##As psf is an array of quantities we cannot use the images['psf'].write() function
# ##all the other arrays do not have quantities. 
# fits.writeto("G300-0_test_psf.fits",images['psf'].data.value,overwrite=True)
# ```
# 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

from gammapy.image import SkyImage
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

# You may see a Warning concerning XSPEC
# As we will note use Xspec spectral models this warning is not important
import sherpa.astro.ui as sh


# In[2]:


# Read the fits file to load them in a sherpa model
hdr = fits.getheader("G300-0_test_counts.fits")
wcs = WCS(hdr)

sh.set_stat("cash")
sh.set_method("simplex")
sh.load_image("G300-0_test_counts.fits")
sh.set_coord("logical")

sh.load_table_model("expo", "G300-0_test_exposure.fits")
sh.load_table_model("bkg", "G300-0_test_background.fits")
sh.load_psf("psf", "G300-0_test_psf.fits")


# In principle one might first want to fit the background amplitude. However the background estimation method already yields the correct normalization, so we freeze the background amplitude to unity instead of adjusting it. The (smoothed) residuals from this background model are then computed and shown.

# In[3]:


sh.set_full_model(bkg)
bkg.ampl = 1
sh.freeze(bkg)

data = sh.get_data_image().y -  sh.get_model_image().y
resid = SkyImage(data=data, wcs=wcs)

resid_table = []  # Keep residual images in a list to show them later
resid_smo6 = resid.smooth(radius = 6)
resid_smo6.plot()
resid_table.append(resid_smo6)


# ### Find and fit the brightest source
# We then find the position of the maximum in the (smoothed) residuals map, and fit a (symmetrical) Gaussian source with that initial position:

# In[4]:


maxcoord = resid_smo6.lookup_max()
maxpix = resid_smo6.wcs_skycoord_to_pixel(maxcoord[0])
sh.set_full_model(bkg + psf(sh.gauss2d.g0) * expo) # creates g0 as a gauss2d instance
g0.xpos = maxpix[0]
g0.ypos = maxpix[1]
sh.freeze(g0.xpos, g0.ypos) # fix the position in the initial fitting step

expo.ampl = 1e-9 # fix exposure amplitude so that typical exposure is of order unity
sh.freeze(expo)
sh.thaw(g0.fwhm, g0.ampl) # in case frozen in a previous iteration

g0.fwhm = 10 # give some reasonable initial values
g0.ampl = maxcoord[1]
sh.fit() # Performs the fit; this takes a little time.


# Fit all parameters of this Gaussian component, fix them and re-compute the residuals map.

# In[5]:


sh.thaw(g0.xpos, g0.ypos)
sh.fit()
sh.freeze(g0)

data = sh.get_data_image().y -  sh.get_model_image().y
resid = SkyImage(data=data, wcs=wcs)

resid_smo6 = resid.smooth(radius = 6)
resid_smo6.show(vmin = -0.5, vmax = 1)
resid_table.append(resid_smo6)


# ### Iteratively find and fit additional sources
# Instantiate additional Gaussian components, and use them to iteratively fit sources, repeating the steps performed above for component g0. (The residuals map is shown after each additional source included in the model.) This takes some time...

# In[6]:


for i in range(1,6):
    sh.create_model_component('gauss2d', 'g' + str(i))

gs = [g0, g1, g2, g3, g4, g5]
sh.set_full_model(bkg + psf(g0+g1+g2+g3+g4+g5) * expo)

for i in range(1, len(gs)) :
    gs[i].ampl = 0   # initialize components with fixed, zero amplitude
    sh.freeze(gs[i])

for i in range(1, len(gs)) :
    maxcoord = resid_smo6.lookup_max()
    maxpix = resid_smo6.wcs_skycoord_to_pixel(maxcoord[0])
    gs[i].xpos = maxpix[0]
    gs[i].ypos = maxpix[1]
    gs[i].fwhm = 10
    gs[i].fwhm = maxcoord[1]

    sh.thaw(gs[i].fwhm)
    sh.thaw(gs[i].ampl)
    sh.fit()

    sh.thaw(gs[i].xpos)
    sh.thaw(gs[i].ypos)
    sh.fit()
    sh.freeze(gs[i])

    data = sh.get_data_image().y -  sh.get_model_image().y # estimate residual map = data - model
    resid = SkyImage(data=data, wcs=wcs)

    resid_smo6 = resid.smooth(radius = 6)
    resid_smo6.show(vmin = -0.5, vmax = 1)
    resid_table.append(resid_smo6)


# ### Generating output table and Test Statistics estimation
# When adding a new source, one need to check the significance of this new source. A frequently used method is the Test Statistics (TS). This is done by comparing the change of statistics when the source is included compared to the null hypothesis (no source ; in practice here we fix the amplitude to zero).
# 
# $TS = Cstat(source) - Cstat(no source)$
# 
# The criterion for a significant source detection is typically that it should improve the test statistic by at least 25 or 30. The last excess fitted (g5) thus not a significant source:

# In[7]:


from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table

rows = []
for idx, g in enumerate(gs):
    ampl = g.ampl.val
    g.ampl = 0
    stati = sh.get_stat_info()[0].statval
    g.ampl = ampl
    statf = sh.get_stat_info()[0].statval
    delstat = stati - statf
    
    coord = resid.wcs_pixel_to_skycoord(g.xpos.val, g.ypos.val)
    pix_scale = resid.wcs_pixel_scale()[0].deg
    sigma = g.fwhm.val * pix_scale * gaussian_fwhm_to_sigma
    rows.append(dict(
        idx=idx,
        delstat=delstat,
        glon=coord.l.deg,
        glat=coord.b.deg,
        sigma=sigma ,
    ))

table = Table(rows=rows, names=rows[0])
table[table['delstat'] > 25]


# In[ ]:


# Small animation to show the source detection at each iteration
from ipywidgets.widgets.interaction import interact

def plot_resid(i):
    fig, ax,cbar = resid_table[i].plot(vmin=-0.5, vmax=1,cmap='CMRmap')
#    ax=plt.gca()
    ax.set_title('CStat=%.2f'%(table['delstat'][i]))
    ax.scatter(
    table['glon'][i], table['glat'][i],
    transform=ax.get_transform('galactic'),
    color='none', edgecolor='azure', marker='o', s=400)
#    plt.savefig('source_%i.png'%(i))
    plt.show()

interact(plot_resid,i=(0,5))


# ## Exercises
# 
# * If you look back to the original image: there's one source that looks like a shell-type supernova remnant.
#     * Try to fit is with a shell morphology model (use ``sh.shell2d('shell')`` to create such a model).
#     * Try to evaluate the ``TS`` and probability of the shell model compared to a Gaussian model hypothesis
#     * You could also try a disk model (use ``sh.disk2d('disk')`` to create one)

# ## What next?
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
