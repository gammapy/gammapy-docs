
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# Set up data directory, target coordinates and max FoV offset. In this example we take 1DC from MSH 15-52

# In[2]:


from gammapy.data import DataStore
import astropy.units as u
from astropy.coordinates import SkyCoord
data_store = DataStore.from_dir('$CTADATA/index/gps')
glat_target = -1.19304588
glon_target = 320.33033806
max_offset = 3 * u.deg


# you can retrieve the coordinates of your source from the web, if you are connected

# In[3]:


#pos_target = SkyCoord.from_name('MSH 15-52')
#pos_target.galactic


# With data_store.obs_table you have a table with the most common per-observation parameters that are used for observation selection. 

# In[4]:


table = data_store.obs_table


# Data selection 
# Using Python / Table methods it is easy to apply any selection you like, always with the goal of making 
# a list or array of OBS_ID, which is then the input to analysis.
# For the current 1DC dataset it is pretty simple, because the only quantities useful for selection are:
#  * pointing position
#  * which irf (i.e. array / zenith angle)
#  
# With real data, there will be more parameters of interest, such as data quality, observation duration, zenith angle, time of observation, ...
# Let's look at one example: select observations that are at offset up to 3 deg from our target

# In[5]:


pos_obs = SkyCoord(table['GLON_PNT'], table['GLAT_PNT'], frame='galactic', unit='deg')
pos_target = SkyCoord(glon_target, glat_target, frame='galactic', unit='deg')
offset = pos_target.separation(pos_obs).deg
mask = (offset < max_offset.value)
table = table[mask]
print('Number of selected observations: ', len(table))


# How to list the OBS_ID

# In[6]:


obs_ids = list(table['OBS_ID'])
print(obs_ids)


# In[7]:


obs_list = data_store.obs_list(obs_ids)
print(obs_list)


# you can visualize the obtained table in your browser

# In[8]:


#table.show_in_browser(jsviewer=True)


# We pick just three runs to speed this notebook up

# In[9]:


obs_ids_now = [110114, 110140, 110893]
print(obs_ids_now)


# In[10]:


obs_list = data_store.obs_list(obs_ids_now)
print(obs_list)


# # Check the pointing positions                                                                                                                
# The grid pointing positions at GLAT = +- 1.2 deg are visible  

# In[11]:


from astropy.coordinates import Angle
plt.scatter(Angle(table['GLON_PNT'], unit='deg'), table['GLAT_PNT'])
plt.xlabel('Galactic longitude (deg)')
plt.ylabel('Galactic latitude (deg)')


# # MAKE SKYMAP
# Define map geometry and an ON region for the image analysis. We use a circle of 0.3deg radius as ON region, knowing that the source is seen as mildly extended (0.12deg) by HESS. 

# In[12]:


from regions import CircleSkyRegion
on_radius = 0.3 * u.deg
on_region = CircleSkyRegion(center=pos_target, radius=on_radius)


# Define reference image centered on the target

# In[13]:


from gammapy.image import SkyImage
xref = pos_target.galactic.l.value
yref = pos_target.galactic.b.value
size = 10 * u.deg
binsz = 0.02 # degree per pixel
npix = int((size / binsz).value)
print(npix)
ref_image = SkyImage.empty(
    nxpix=500, nypix=500, binsz=binsz,
    xref=xref, yref=yref,
    proj='CAR', coordsys='GAL',
)
print(ref_image)


# We use the ring background estimation method, and an exclusion mask that excludes the bright source at the Galactic center.

# In[14]:


exclusion_mask = ref_image.region_mask(on_region)
exclusion_mask.data = 1 - exclusion_mask.data
exclusion_mask.plot()


# In[15]:


from gammapy.background import RingBackgroundEstimator
bkg_estimator = RingBackgroundEstimator(
    r_in=0.6 * u.deg,
    width=0.2 * u.deg,
)


# In[16]:


from gammapy.image import IACTBasicImageEstimator
image_estimator = IACTBasicImageEstimator(
    reference=ref_image,
    emin=200 * u.GeV,
    emax=100 * u.TeV,
    offset_max=3 * u.deg,
    background_estimator=bkg_estimator,
    exclusion_mask=exclusion_mask,
)


# The image_estimator contains 6 images. How to know which images it contains

# In[17]:


images = image_estimator.run(obs_list)


# In[18]:


print(images)


# In[19]:


print(images.names)


# How to show the images. The counts map

# In[20]:


cts_image = images['counts']
cts_image.show(vmin=0,vmax=10,add_cbar=True)


# In[21]:


cts_image_cutoff = images['counts'].cutout(position=pos_target,size=(3*u.deg, 3*u.deg),)


# In[22]:


cts_image_cutoff.show(vmin=0,vmax=10,add_cbar=True)


# In[23]:


cts_image_cutoff.smooth(radius=2).show(vmin=0,vmax=10,add_cbar=True)
#shift tab for documentation


# In[24]:


bkg_image = images['background']
bkg_image.show(vmin=0,vmax=4,add_cbar=True)


# In[25]:


excess_image = images['excess']
excess_image.show(vmin=0,vmax=5,add_cbar=True)


# In[26]:


excess_image.smooth(radius=2).show(vmin=0,vmax=5,add_cbar=True)


# In[27]:


excess_image_cutoff = excess_image.cutout(position=pos_target,size=(3*u.deg, 3*u.deg),)


# In[28]:


excess_image_cutoff.smooth(radius=2).show(vmin=0,vmax=4,add_cbar=True)


# Save the images in fits files to be used later on for an eventual sherpa fitting

# In[29]:


from astropy.io import fits
images['counts'].write("../datasets/images/MSH15-52_counts.fits.gz", clobber=True)
images['background'].write("../datasets/images/MSH15-52_background.fits.gz", clobber=True)
images['exposure'].write("../datasets/images/MSH15-52_exposure.fits.gz", clobber=True)
#As psf is an array of quantities we cannot use the images['psf'].write() function
fits.writeto("../datasets/images/MSH15-52_psf.fits.gz",images['psf'].data.value,overwrite=True)


# Let's compute the significance map by using the Li&Ma significance definition

# In[30]:


from astropy.convolution import Tophat2DKernel
kernel = Tophat2DKernel(4)
from gammapy.detect import compute_lima_image
lima_image = compute_lima_image(cts_image,bkg_image,kernel)['significance']


# In[31]:


lima_image.show(vmin=0,vmax=10,add_cbar=True)


# Let's compute a TS image instead (1D likelihood fit of the source amplitude)

# In[32]:


# cut out smaller piece of the PSF image to save computing time
# for covenience we're "misusing" the SkyImage class represent the PSF on the sky.
kernel = images['psf'].cutout(pos_target, size= 1.1 * u.deg)
kernel.show()


# In[33]:


from gammapy.detect import TSImageEstimator
ts_image_estimator = TSImageEstimator()
images_ts = ts_image_estimator.run(images, kernel.data)
print(images_ts.names)


# We run a peak finder on the sqrt_ts image to get a list of sources (positions and peak sqrt_ts values).

# In[34]:


from photutils.detection import find_peaks
sources = find_peaks(data=images_ts['sqrt_ts'].data, threshold=5, wcs=images_ts['sqrt_ts'].wcs)
sources


# In[35]:


images_ts['sqrt_ts'].cutout(
    position=SkyCoord(pos_target, unit='deg', frame='galactic'),
    size=(3*u.deg, 3*u.deg)).show(add_cbar=True,vmax=10)


# # Spectral analysis
# We’ll run a spectral analysis using the classical reflected regions background estimation method, and using the on-off (often called WSTAT) likelihood function.

# In[36]:


from gammapy.spectrum import (
    SpectrumExtraction,
    SpectrumFit,
    SpectrumResult,
    models,
    SpectrumEnergyGroupMaker,
    FluxPointEstimator,
)


# Next we will manually perform a background estimate by placing reflected regions around the pointing position and looking at the source statistics. This will result in a gammapy.background.BackgroundEstimate that serves as input for other classes in gammapy.

# In[37]:


from gammapy.background import ReflectedRegionsBackgroundEstimator
bkg_estimator = ReflectedRegionsBackgroundEstimator(
    obs_list=obs_list,
    on_region=on_region,
    exclusion_mask=exclusion_mask,
)


# In[38]:


bkg_estimator.run()
bkg_estimate = bkg_estimator.result
print(bkg_estimate[0])


# In[39]:


bkg_estimator.plot()


# we’re going to look at the overall source statistics in our signal region.

# In[40]:


from gammapy.data import ObservationStats, ObservationSummary
stats = []
for obs, bkg in zip(obs_list, bkg_estimator.result):
    stats.append(ObservationStats.from_obs(obs, bkg))
print(stats[0],stats[1],stats[2])    


# In[41]:


obs_summary = ObservationSummary(stats)
obs_summary.plot_excess_vs_livetime()


# In[42]:


obs_summary.plot_significance_vs_livetime()


# Now, we’re going to extract a spectrum using the SpectrumExtraction class. 
# Thus, we instantiate a SpectrumExtraction object that will do the extraction. 

# In[43]:


extract = SpectrumExtraction(
    obs_list=obs_list,
    bkg_estimate=bkg_estimate,
)

extract.run()


# Now we will look at the files we just created. We will use the SpectrumObservation object that are still in memory from the extraction step. Note, however, that you could also read them from disk if you have written them in the step above . The ANALYSIS_DIR folder contains 4 FITS files for each observation. These files are described in detail at https://gamma-astro-data-formats.readthedocs.io/en/latest/ogip/index.html. In short they correspond to the on vector, the off vector, the effectie area, and the energy dispersion.

# In[44]:


extract.observations[0].peek()


# Now we’ll fit a global model to the spectrum. First we do a joint likelihood fit to all observations. 

# In[45]:


model = models.PowerLaw(
    index = 2 * u.Unit(''),
    amplitude = 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
    reference = 1 * u.TeV,
)

fit = SpectrumFit(extract.observations, model,
                 fit_range=(1*u.TeV, 10*u.TeV))
#probably not working
fit.fit()
fit.est_errors()
print(fit.result[0])


# We will also produce a debug plot in order to show how the global fit matches one of the individual observations.

# In[46]:


fit.result[0].plot()


# we can compute flux points by fitting the norm of the global model in energy bands. We’ll use a fixed energy binning for now.

# In[47]:


from gammapy.utils.energy import EnergyBounds
# Flux points are computed on stacked observation
stacked_obs = extract.observations.stack()
print(stacked_obs)

ebounds = EnergyBounds.equal_log_spacing(0.1, 40, 15, unit = u.TeV)

seg = SpectrumEnergyGroupMaker(obs=stacked_obs)
seg.compute_range_safe()
seg.compute_groups_fixed(ebounds=ebounds)

print(seg.groups)


# In[48]:


fpe = FluxPointEstimator(
    obs=stacked_obs,
    groups=seg.groups,
    model=fit.result[0].model,
)
fpe.compute_points()
print(fpe.flux_points.table)


# In[49]:


fpe.flux_points.plot()


# We try to compare the obtained spectral points and spectral results with the input model in the xml file. 
# So far there is no possibility to read xml files from gammapy

# In[50]:


get_ipython().system("  grep 'MSH' $CTADATA/models/models_gps.xml -A10")


# We plug in the spectral parameters of the input model in a gammapy spectral function.

# In[51]:


spec_true = models.PowerLaw(
    index = 2.2699 * u.Unit(''),
    amplitude = 569.99999e-20 * u.Unit('cm-2 s-1 MeV-1'),
    reference = 1 * u.TeV,
)


# we plot the obtained spectrum with the one in input together

# In[52]:


opts = dict(
    energy_range = [0.08, 50] * u.TeV,
    energy_power=2,
    flux_unit='erg-1 cm-2 s-1',
)

total_result = SpectrumResult(
    model=fit.result[0].model,
    points=fpe.flux_points,
)


ax_sed, ax_resid = total_result.plot(
    fig_kwargs=dict(figsize=(8,8)),
    point_kwargs=dict(color='blue', label='Measured'),
    **opts
)

spec_true.plot(
    ax=ax_sed,
    label='True',
    color='magenta',
    **opts
)
ax_sed.legend()


# # Sherpa Morphological fit

# In[53]:


import sherpa.astro.ui as sh
sh.set_stat("cash")
sh.set_method("simplex")
sh.load_image('../datasets/images/MSH15-52_counts.fits.gz')
sh.set_coord("logical")

sh.load_table_model("expo", "../datasets/images/MSH15-52_exposure.fits.gz")
sh.load_table_model("bkg", "../datasets/images/MSH15-52_background.fits.gz")
sh.load_psf("psf", "../datasets/images/MSH15-52_psf.fits.gz")


# In[54]:


sh.set_full_model(bkg)
bkg.ampl = 1
sh.freeze(bkg)

data = sh.get_data_image().y -  sh.get_model_image().y
resid = SkyImage(data=data, wcs=ref_image.wcs)

resid_table=[]  #Keep residual images in a list to show them later
resid_smo6 = resid.smooth(radius = 6)
resid_smo6.plot(vmax=5,add_cbar=True)
resid_table.append(resid_smo6)


# In[55]:


maxcoord = resid_smo6.lookup_max()
maxpix = resid_smo6.wcs_skycoord_to_pixel(maxcoord[0])
print(maxcoord)
print(maxpix)
sh.set_full_model(bkg + psf(sh.gauss2d.g0) * expo) # creates g0 as a gauss2d instance


# In[56]:


g0.xpos = maxpix[0]
g0.ypos = maxpix[1]
sh.freeze(g0.xpos, g0.ypos) 
expo.ampl = 1e-9
sh.freeze(expo)
sh.thaw(g0.fwhm, g0.ampl)
g0.fwhm = 10
g0.ampl = maxcoord[1]
sh.fit()


# In[57]:


sh.thaw(g0.xpos, g0.ypos)
sh.fit()
sh.covar()
sh.freeze(g0)

data = sh.get_data_image().y -  sh.get_model_image().y
resid = SkyImage(data=data, wcs=ref_image.wcs)

resid_smo6 = resid.smooth(radius = 6)
resid_smo6.show(vmin = -0.5, vmax = 1,add_cbar=True)
resid_table.append(resid_smo6)


# In[58]:


from astropy.stats import gaussian_fwhm_to_sigma
coord = resid.wcs_pixel_to_skycoord(g0.xpos.val, g0.ypos.val)
pix_scale = resid.wcs_pixel_scale()[0].deg
sigma = g0.fwhm.val * pix_scale * gaussian_fwhm_to_sigma
print('the Sigma is: ', sigma, ' deg')


# In[59]:


print(coord)


# In[60]:


sh.covar()

