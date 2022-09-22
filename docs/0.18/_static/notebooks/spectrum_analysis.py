#!/usr/bin/env python
# coding: utf-8

# # Spectral analysis with Gammapy

# ## Prerequisites 
# 
# - Understanding how spectral extraction is performed in Cherenkov astronomy, in particular regarding OFF background measurements. 
# - Understanding the basics data reduction and modeling/fitting process with the gammapy library API as shown in the [first gammapy analysis with the gammapy library API tutorial](analysis_2.ipynb)
# 
# ## Context
# 
# While 3D analysis allows in principle to deal with complex situations such as overlapping sources, in many cases, it is not required to extract the spectrum of a source. Spectral analysis, where all data inside a ON region are binned into 1D datasets, provides a nice alternative. 
# 
# In classical Cherenkov astronomy, it is used with a specific background estimation technique that relies on OFF measurements taken in the field-of-view in regions where the background
# rate is assumed to be equal to the one in the ON region. 
# 
# This allows to use a specific fit statistics for ON-OFF measurements, the wstat (see `~gammapy.stats.fit_statistics`), where no background model is assumed. Background is treated as a set of nuisance parameters. This removes some systematic effects connected
# to the choice or the quality of the background model. But this comes at the expense of larger statistical uncertainties on the fitted model parameters.
# 
# **Objective: perform a full region based spectral analysis of 4 Crab observations of  H.E.S.S. data release 1 and fit the resulting datasets.**
# 
# ## Introduction
# 
# Here, as usual, we use the `~gammapy.data.DataStore` to retrieve a list of selected observations (`~gammapy.data.Observations`). Then, we define the ON region containing the source and the geometry of the `~gammapy.datasets.SpectrumDataset` object we want to produce. We then create the corresponding dataset Maker. 
# 
# We have to define the Maker object that will extract the OFF counts from reflected regions in the field-of-view. To ensure we use data in an energy range where the quality of the IRFs is good enough we also create a safe range Maker.
# 
# We can then proceed with data reduction with a loop over all selected observations to produce datasets in the relevant geometry.
# 
# We can then explore the resulting datasets and look at the cumulative signal and significance of our source. We finally proceed with model fitting. 
# 
# In practice, we have to:
# - Create a `~gammapy.data.DataStore` poiting to the relevant data 
# - Apply an observation selection to produce a list of observations, a `~gammapy.data.Observations` object.
# - Define a geometry of the spectrum we want to produce:
#     - Create a `~regions.CircleSkyRegion` for the ON extraction region
#     - Create a `~gammapy.maps.MapAxis` for the energy binnings: one for the reconstructed (i.e. measured) energy, the other for the true energy (i.e. the one used by IRFs and models)
# - Create the necessary makers : 
#     - the spectrum dataset maker : `~gammapy.makers.SpectrumDatasetMaker`
#     - the OFF background maker, here a `~gammapy.makers.ReflectedRegionsBackgroundMaker`
#     - and the safe range maker : `~gammapy.makers.SafeRangeMaker`
# - Perform the data reduction loop. And for every observation:
#     - Apply the makers sequentially to produce a `~gammapy.datasets.SpectrumDatasetOnOff`
#     - Append it to list of datasets
# - Define the `~gammapy.modeling.models.SkyModel` to apply to the dataset.
# - Create a `~gammapy.modeling.Fit` object and run it to fit the model parameters
# - Apply a `~gammapy.estimators.FluxPointsEstimator` to compute flux points for the spectral part of the fit.
# 

# ## Setup
# 
# As usual, we'll start with some setup ...

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


# Check package versions
import gammapy
import numpy as np
import astropy
import regions

print("gammapy:", gammapy.__version__)
print("numpy:", np.__version__)
print("astropy", astropy.__version__)
print("regions", regions.__version__)


# In[ ]:


from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from gammapy.maps import Map, MapAxis
from gammapy.modeling import Fit
from gammapy.data import DataStore
from gammapy.datasets import (
    Datasets,
    SpectrumDataset,
    SpectrumDatasetOnOff,
    FluxPointsDataset,
)
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    create_crab_spectral_model,
    SkyModel,
)
from gammapy.makers import (
    SafeMaskMaker,
    SpectrumDatasetMaker,
    ReflectedRegionsBackgroundMaker,
)
from gammapy.estimators import FluxPointsEstimator
from gammapy.visualization import plot_spectrum_datasets_off_regions


# ## Load Data
# 
# First, we select and load some H.E.S.S. observations of the Crab nebula (simulated events for now).
# 
# We will access the events, effective area, energy dispersion, livetime and PSF for containement correction.

# In[ ]:


datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
obs_ids = [23523, 23526, 23559, 23592]
observations = datastore.get_observations(obs_ids)


# ## Define Target Region
# 
# The next step is to define a signal extraction region, also known as on region. In the simplest case this is just a [CircleSkyRegion](http://astropy-regions.readthedocs.io/en/latest/api/regions.CircleSkyRegion.html), but here we will use the ``Target`` class in gammapy that is useful for book-keeping if you run several analysis in a script.

# In[ ]:


target_position = SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs")
on_region_radius = Angle("0.11 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)


# ## Create exclusion mask
# 
# We will use the reflected regions method to place off regions to estimate the background level in the on region.
# To make sure the off regions don't contain gamma-ray emission, we create an exclusion mask.
# 
# Using http://gamma-sky.net/ we find that there's only one known gamma-ray source near the Crab nebula: the AGN called [RGB J0521+212](http://gamma-sky.net/#/cat/tev/23) at GLON = 183.604 deg and GLAT = -8.708 deg.

# In[ ]:


exclusion_region = CircleSkyRegion(
    center=SkyCoord(183.604, -8.708, unit="deg", frame="galactic"),
    radius=0.5 * u.deg,
)

skydir = target_position.galactic
exclusion_mask = Map.create(
    npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", frame="icrs"
)

mask = exclusion_mask.geom.region_mask([exclusion_region], inside=False)
exclusion_mask.data = mask
exclusion_mask.plot();


# ## Run data reduction chain
# 
# We begin with the configuration of the maker classes:

# In[ ]:


e_reco = MapAxis.from_energy_bounds(0.1, 40, 40, unit="TeV", name="energy")
e_true = MapAxis.from_energy_bounds(
    0.05, 100, 200, unit="TeV", name="energy_true"
)
dataset_empty = SpectrumDataset.create(
    e_reco=e_reco, e_true=e_true, region=on_region
)


# In[ ]:


dataset_maker = SpectrumDatasetMaker(
    containment_correction=False, selection=["counts", "exposure", "edisp"]
)
bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'datasets = Datasets()\n\nfor obs_id, observation in zip(obs_ids, observations):\n    dataset = dataset_maker.run(\n        dataset_empty.copy(name=str(obs_id)), observation\n    )\n    dataset_on_off = bkg_maker.run(dataset, observation)\n    dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)\n    datasets.append(dataset_on_off)')


# ## Plot off regions

# In[ ]:


plt.figure(figsize=(8, 8))
_, ax, _ = exclusion_mask.plot()
on_region.to_pixel(ax.wcs).plot(ax=ax, edgecolor="k")
plot_spectrum_datasets_off_regions(ax=ax, datasets=datasets)


# ## Source statistic
# 
# Next we're going to look at the overall source statistics in our signal region.

# In[ ]:


info_table = datasets.info_table(cumulative=True)


# In[ ]:


info_table


# In[ ]:


plt.plot(
    info_table["livetime"].to("h"), info_table["excess"], marker="o", ls="none"
)
plt.xlabel("Livetime [h]")
plt.ylabel("Excess");


# In[ ]:


plt.plot(
    info_table["livetime"].to("h"),
    info_table["sqrt_ts"],
    marker="o",
    ls="none",
)
plt.xlabel("Livetime [h]")
plt.ylabel("Sqrt(TS)");


# Finally you can write the extrated datasets to disk using the OGIP format (PHA, ARF, RMF, BKG, see [here](https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/ogip/index.html) for details):

# In[ ]:


path = Path("spectrum_analysis")
path.mkdir(exist_ok=True)


# In[ ]:


for dataset in datasets:
    dataset.to_ogip_files(outdir=path, overwrite=True)


# If you want to read back the datasets from disk you can use:

# In[ ]:


datasets = Datasets()
for obs_id in obs_ids:
    filename = path / f"pha_obs{obs_id}.fits"
    datasets.append(SpectrumDatasetOnOff.from_ogip_files(filename))


# ## Fit spectrum
# 
# Now we'll fit a global model to the spectrum. First we do a joint likelihood fit to all observations. If you want to stack the observations see below. We will also produce a debug plot in order to show how the global fit matches one of the individual observations.

# In[ ]:


spectral_model = PowerLawSpectralModel(
    index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
)
model = SkyModel(spectral_model=spectral_model, name="crab")

for dataset in datasets:
    dataset.models = model

fit_joint = Fit(datasets)
result_joint = fit_joint.run()

# we make a copy here to compare it later
model_best_joint = model.copy()


# ## Fit quality and model residuals

# We can access the results dictionary to see if the fit converged:

# In[ ]:


print(result_joint)


# A simple way to inspect the model residuals is using the function `~SpectrumDataset.plot_fit()`

# In[ ]:


ax_spectrum, ax_residuals = datasets[0].plot_fit()
ax_spectrum.set_ylim(0.1, 40)


# For more ways of assessing fit quality, please refer to the dedicated [modeling and fitting tutorial](modeling.ipynb).

# ## Compute Flux Points
# 
# To round up our analysis we can compute flux points by fitting the norm of the global model in energy bands. We'll use a fixed energy binning for now:

# In[ ]:


e_min, e_max = 0.7, 30
energy_edges = np.logspace(np.log10(e_min), np.log10(e_max), 11) * u.TeV


# Now we create an instance of the `~gammapy.estimators.FluxPointsEstimator`, by passing the dataset and the energy binning:

# In[ ]:


fpe = FluxPointsEstimator(energy_edges=energy_edges, source="crab")
flux_points = fpe.run(datasets=datasets)


# Here is a the table of the resulting flux points:

# In[ ]:


flux_points.table_formatted


# Now we plot the flux points and their likelihood profiles. For the plotting of upper limits we choose a threshold of TS < 4.

# In[ ]:


plt.figure(figsize=(8, 5))
flux_points.table["is_ul"] = flux_points.table["ts"] < 4
ax = flux_points.plot(
    energy_power=2, flux_unit="erg-1 cm-2 s-1", color="darkorange"
)
flux_points.to_sed_type("e2dnde").plot_ts_profiles(ax=ax);


# The final plot with the best fit model, flux points and residuals can be quickly made like this: 

# In[ ]:


flux_points_dataset = FluxPointsDataset(
    data=flux_points, models=model_best_joint
)


# In[ ]:


flux_points_dataset.plot_fit();


# ## Stack observations
# 
# And alternative approach to fitting the spectrum is stacking all observations first and the fitting a model. For this we first stack the individual datasets:

# In[ ]:


dataset_stacked = Datasets(datasets).stack_reduce()


# Again we set the model on the dataset we would like to fit (in this case it's only a single one) and pass it to the `~gammapy.modeling.Fit` object:

# In[ ]:


dataset_stacked.models = model
stacked_fit = Fit([dataset_stacked])
result_stacked = stacked_fit.run()

# make a copy to compare later
model_best_stacked = model.copy()


# In[ ]:


print(result_stacked)


# In[ ]:


model_best_joint.parameters.to_table()


# In[ ]:


model_best_stacked.parameters.to_table()


# Finally, we compare the results of our stacked analysis to a previously published Crab Nebula Spectrum for reference. This is available in `~gammapy.modeling.models.create_crab_spectral_model`.

# In[ ]:


plot_kwargs = {
    "energy_range": [0.1, 30] * u.TeV,
    "energy_power": 2,
    "flux_unit": "erg-1 cm-2 s-1",
}

# plot stacked model
model_best_stacked.spectral_model.plot(
    **plot_kwargs, label="Stacked analysis result"
)
model_best_stacked.spectral_model.plot_error(**plot_kwargs)

# plot joint model
model_best_joint.spectral_model.plot(
    **plot_kwargs, label="Joint analysis result", ls="--"
)
model_best_joint.spectral_model.plot_error(**plot_kwargs)

create_crab_spectral_model("hess_pl").plot(
    **plot_kwargs, label="Crab reference"
)
plt.legend()


# ## Exercises
# 
# Now you have learned the basics of a spectral analysis with Gammapy. To practice you can continue with the following exercises:
# 
# - Fit a different spectral model to the data.
#   You could try `~gammapy.modeling.models.ExpCutoffPowerLawSpectralModel` or `~gammapy.modeling.models.LogParabolaSpectralModel`.
# - Compute flux points for the stacked dataset.
# - Create a `~gammapy.estimators.FluxPointsDataset` with the flux points you have computed for the stacked dataset and fit the flux points again with obe of the spectral models. How does the result compare to the best fit model, that was directly fitted to the counts data?

# ## What next?
# 
# The methods shown in this tutorial is valid for point-like or midly extended sources where we can assume that the IRF taken at the region center is valid over the whole region. If one wants to extract the 1D spectrum of a large source and properly average the response over the extraction region, one has to use a different approach explained in [the extended source spectral analysis tutorial](extended_source_spectral_analysis.ipynb).

# In[ ]:




