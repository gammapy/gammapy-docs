
# coding: utf-8

# # Dark matter utilities
# 
# ## Introduction 
# 
# Gammapy has some convenience methods for dark matter analyses in [gammapy.astro.darkmatter](http://docs.gammapy.org/0.8/astro/darkmatter/index.html). These include J-Factor computation and calculation the expected gamma flux for a number of annihilation channels. They are presented in this notebook. 
# 
# The basic concepts of indirect dark matter searches, however, are not explained. So this is aimed at people who already know what the want to do. A good introduction to indirect dark matter searches is given for example in https://arxiv.org/pdf/1012.4515.pdf (Chapter 1 and 5)

# ## Setup
# 
# As always, we start with some setup for the notebook, and with imports.

# In[ ]:


from gammapy.astro.darkmatter import (
    profiles,
    JFactory,
    PrimaryFlux,
    compute_dm_flux,
)

from gammapy.maps import WcsGeom, WcsNDMap
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm
from regions import CircleSkyRegion
import astropy.units as u
import numpy as np


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## Profiles
# 
# The following dark matter profiles are currently implemented. Each model can be scaled to a given density at a certain distance. These parameters are controlled by ``profiles.DMProfile.LOCAL_DENSITY`` and ``profiles.DMProfile.DISTANCE_GC``

# In[ ]:


profiles.DMProfile.__subclasses__()


# In[ ]:


for profile in profiles.DMProfile.__subclasses__():
    p = profile()
    p.scale_to_local_density()
    radii = np.logspace(-3, 2, 100) * u.kpc
    plt.plot(radii, p(radii), label=p.__class__.__name__)

plt.loglog()
plt.axvline(8.5, linestyle="dashed", color="black", label="local density")
plt.legend()

print(
    "The assumed local density is {} at a distance to the GC of {}".format(
        profiles.DMProfile.LOCAL_DENSITY, profiles.DMProfile.DISTANCE_GC
    )
)


# ## J Factors
# 
# There are utilies to compute J-Factor maps can can serve as a basis to compute J-Factors for certain regions. In the following we compute a J-Factor map for the Galactic Centre region

# In[ ]:


profile = profiles.NFWProfile()

# Adopt standard values used in HESS
profiles.DMProfile.DISTANCE_GC = 8.5 * u.kpc
profiles.DMProfile.LOCAL_DENSITY = 0.39 * u.Unit("GeV / cm3")

profile.scale_to_local_density()

position = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
geom = WcsGeom.create(binsz=0.05, skydir=position, width=3.0, coordsys="GAL")


# In[ ]:


jfactory = JFactory(
    geom=geom, profile=profile, distance=profiles.DMProfile.DISTANCE_GC
)
jfact = jfactory.compute_jfactor()


# In[ ]:


jfact_map = WcsNDMap(geom=geom, data=jfact.value, unit=jfact.unit)
fig, ax, im = jfact_map.plot(cmap="viridis", norm=LogNorm(), add_cbar=True)
plt.title("J-Factor [{}]".format(jfact_map.unit))

# 1 deg circle usually used in H.E.S.S. analyses
sky_reg = CircleSkyRegion(center=position, radius=1 * u.deg)
pix_reg = sky_reg.to_pixel(wcs=geom.wcs)
pix_reg.plot(ax=ax, facecolor="none", edgecolor="red", label="1 deg circle")
plt.legend()


# In[ ]:


# NOTE: https://arxiv.org/abs/1607.08142 cite 2.67e21 without the the +/- 0.3 deg band around the plane
mask = pix_reg.to_mask()
data = mask.multiply(jfact.data)
total_jfact = np.sum(data)
print(
    "J-factor in 1 deg circle around GC assuming a "
    "{} is {:.3g}".format(profile.__class__.__name__, total_jfact)
)


# ## Gamma-ray spectra at production
# 
# The gamma-ray spectrum per annihilation is a further ingredient for a dark matter analysis. The following annihilation channels are supported. For more info see https://arxiv.org/pdf/1012.4515.pdf

# In[ ]:


fluxes = PrimaryFlux(mDM="1 TeV", channel="eL")
print(fluxes.allowed_channels)


# In[ ]:


fig, axes = plt.subplots(4, 1, figsize=(6, 16))
mDMs = [0.01, 0.1, 1, 10] * u.TeV

for mDM, ax in zip(mDMs, axes):
    fluxes.mDM = mDM
    ax.set_title(r"m$_{{\mathrm{{DM}}}}$ = {}".format(mDM))
    ax.set_yscale("log")
    ax.set_ylabel("dN/dE")

    for channel in ["tau", "mu", "b", "Z"]:
        fluxes.channel = channel
        fluxes.table_model.plot(
            energy_range=[mDM / 100, mDM],
            ax=ax,
            label=channel,
            flux_unit="1/GeV",
        )

axes[0].legend()
plt.subplots_adjust(hspace=0.5)


# ## Flux maps
# 
# Finally flux maps can be produced like this:

# In[ ]:


flux = compute_dm_flux(
    jfact=jfact,
    prim_flux=fluxes,
    x_section="1e-26 cm3s-1",
    energy_range=[0.1, 10] * u.TeV,
)


# In[ ]:


flux_map = WcsNDMap(geom=geom, data=flux.value, unit=flux.unit)

fig, ax, im = flux_map.plot(cmap="viridis", norm=LogNorm(), add_cbar=True)
plt.title(
    "Flux [{}]\n m$_{{DM}}$={}, channel={}".format(
        flux_map.unit, fluxes.mDM.to("TeV"), fluxes.channel
    )
);

