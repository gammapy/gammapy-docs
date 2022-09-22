#!/usr/bin/env python
# coding: utf-8

# # 3D analysis
# 
# This tutorial does a 3D map based analsis on the galactic center, using simulated observations from the CTA-1DC. We will use the high level interface for the data reduction, and then do a detailed modelling. This will be done in two different ways:
# 
# - stacking all the maps together and fitting the stacked maps
# - handling all the observations separately and doing a joint fitting on all the maps

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from pathlib import Path
from regions import CircleSkyRegion
from scipy.stats import norm
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling.models import (
    SkyModel,
    ExpCutoffPowerLawSpectralModel,
    PointSpatialModel,
    FoVBackgroundModel,
    Models,
)
from gammapy.modeling import Fit
from gammapy.maps import Map
from gammapy.estimators import ExcessMapEstimator
from gammapy.datasets import MapDataset


# ## Analysis configuration

# In this section we select observations and define the analysis geometries, irrespective of  joint/stacked analysis. For configuration of the analysis, we will programatically build a config file from scratch.

# In[ ]:


config = AnalysisConfig()
# The config file is now empty, with only a few defaults specified.
print(config)


# In[ ]:


# Selecting the observations
config.observations.datastore = "$GAMMAPY_DATA/cta-1dc/index/gps/"
config.observations.obs_ids = [110380, 111140, 111159]


# In[ ]:


# Defining a reference geometry for the reduced datasets

config.datasets.type = "3d"  # Analysis type is 3D

config.datasets.geom.wcs.skydir = {
    "lon": "0 deg",
    "lat": "0 deg",
    "frame": "galactic",
}  # The WCS geometry - centered on the galactic center
config.datasets.geom.wcs.fov = {"width": "10 deg", "height": "8 deg"}
config.datasets.geom.wcs.binsize = "0.02 deg"

# The FoV radius to use for cutouts
config.datasets.geom.selection.offset_max = 3.5 * u.deg
config.datasets.safe_mask.methods = ["aeff-default", "offset-max"]

# We now fix the energy axis for the counts map - (the reconstructed energy binning)
config.datasets.geom.axes.energy.min = "0.1 TeV"
config.datasets.geom.axes.energy.max = "10 TeV"
config.datasets.geom.axes.energy.nbins = 10

# We now fix the energy axis for the IRF maps (exposure, etc) - (the true enery binning)
config.datasets.geom.axes.energy_true.min = "0.08 TeV"
config.datasets.geom.axes.energy_true.max = "12 TeV"
config.datasets.geom.axes.energy_true.nbins = 14


# In[ ]:


print(config)


# ## Configuration for stacked and joint analysis
# 
# This is done just by specfiying the flag on `config.datasets.stack`. Since the internal machinery will work differently for the two cases, we will write it as two config files and save it to disc in YAML format for future reference. 

# In[ ]:


config_stack = config.copy(deep=True)
config_stack.datasets.stack = True

config_joint = config.copy(deep=True)
config_joint.datasets.stack = False


# In[ ]:


# To prevent unnecessary cluttering, we write it in a separate folder.
path = Path("analysis_3d")
path.mkdir(exist_ok=True)
config_joint.write(path=path / "config_joint.yaml", overwrite=True)
config_stack.write(path=path / "config_stack.yaml", overwrite=True)


# ## Stacked analysis
# 
# ### Data reduction
# 
# We first show the steps for the stacked analysis and then repeat the same for the joint analysis later
# 

# In[ ]:


# Reading yaml file:
config_stacked = AnalysisConfig.read(path=path / "config_stack.yaml")


# In[ ]:


analysis_stacked = Analysis(config_stacked)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# select observations:\nanalysis_stacked.get_observations()\n\n# run data reduction\nanalysis_stacked.get_datasets()')


# We have one final dataset, which you can print and explore

# In[ ]:


dataset_stacked = analysis_stacked.datasets["stacked"]
print(dataset_stacked)


# In[ ]:


# To plot a smooth counts map
dataset_stacked.counts.smooth(0.02 * u.deg).plot_interactive(add_cbar=True)


# In[ ]:


# And the background map
dataset_stacked.background.plot_interactive(add_cbar=True)


# In[ ]:


# You can also get an excess image with a few lines of code:
excess = dataset_stacked.excess.sum_over_axes()
excess.smooth("0.06 deg").plot(stretch="sqrt", add_cbar=True);


# ### Modeling and fitting
# 
# Now comes the interesting part of the analysis - choosing appropriate models for our source and fitting them.
# 
# We choose a point source model with an exponential cutoff power-law spectrum.
# 
# To select a certain energy range for the fit we can create a fit mask:

# In[ ]:


coords = dataset_stacked.counts.geom.get_coord()
mask_energy = coords["energy"] > 0.3 * u.TeV
dataset_stacked.mask_fit = Map.from_geom(
    geom=dataset_stacked.counts.geom, data=mask_energy
)


# In[ ]:


spatial_model = PointSpatialModel(
    lon_0="-0.05 deg", lat_0="-0.05 deg", frame="galactic"
)
spectral_model = ExpCutoffPowerLawSpectralModel(
    index=2.3,
    amplitude=2.8e-12 * u.Unit("cm-2 s-1 TeV-1"),
    reference=1.0 * u.TeV,
    lambda_=0.02 / u.TeV,
)

model = SkyModel(
    spatial_model=spatial_model,
    spectral_model=spectral_model,
    name="gc-source",
)

bkg_model = FoVBackgroundModel(dataset_name="stacked")
bkg_model.spectral_model.norm.value = 1.3

models_stacked = Models([model, bkg_model])

dataset_stacked.models = models_stacked


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit = Fit([dataset_stacked])\nresult = fit.run(optimize_opts={"print_level": 1})')


# ### Fit quality assesment and model residuals for a `MapDataset`

# We can access the results dictionary to see if the fit converged:

# In[ ]:


print(result)


# Check best-fit parameters and error estimates:

# In[ ]:


result.parameters.to_table()


# A quick way to inspect the model residuals is using the function `~MapDataset.plot_residuals_spatial()`. This function computes and plots a residual image (by default, the smoothing radius is `0.1 deg` and `method=diff`, which corresponds to a simple `data - model` plot):

# In[ ]:


dataset_stacked.plot_residuals_spatial(
    method="diff/sqrt(model)", vmin=-1, vmax=1
);


# The more general function `~MapDataset.plot_residuals()` can also extract and display spectral residuals in a region:

# In[ ]:


region = CircleSkyRegion(spatial_model.position, radius=0.15 * u.deg)

dataset_stacked.plot_residuals(
    kwargs_spatial=dict(method="diff/sqrt(model)", vmin=-1, vmax=1),
    kwargs_spectral=dict(region=region),
);


# This way of accessing residuals is quick and handy, but comes with limitations. For example:
# - In case a fitting energy range was defined using a `MapDataset.mask_fit`, it won't be taken into account. Residuals will be summed up over the whole reconstructed energy range
# - In order to make a proper statistic treatment, instead of simple residuals a proper residuals significance map should be computed
# 
# A more accurate way to inspect spatial residuals is the following:

# In[ ]:


estimator = ExcessMapEstimator(
    correlation_radius="0.1 deg",
    selection_optional=[],
    energy_edges=[0.1, 1, 10] * u.TeV,
)

result = estimator.run(dataset_stacked)

result["sqrt_ts"].plot_grid(
    figsize=(12, 4), cmap="coolwarm", add_cbar=True, vmin=-5, vmax=5, ncols=2
);


# Distribution of residuals significance in the full map geometry:

# In[ ]:


# TODO: clean this up
significance_data = result["sqrt_ts"].data

# #Remove bins that are inside an exclusion region, that would create an artificial peak at significance=0.
# #For now these lines are useless, because to_image() drops the mask fit
# mask_data = dataset_image.mask_fit.sum_over_axes().data
# excluded = mask_data == 0
# significance_data = significance_data[~excluded]
selection = np.isfinite(significance_data) & ~(significance_data == 0)
significance_data = significance_data[selection]

plt.hist(significance_data, density=True, alpha=0.9, color="red", bins=40)
mu, std = norm.fit(significance_data)

x = np.linspace(-5, 5, 100)
p = norm.pdf(x, mu, std)

plt.plot(
    x,
    p,
    lw=2,
    color="black",
    label=r"$\mu$ = {:.2f}, $\sigma$ = {:.2f}".format(mu, std),
)
plt.legend(fontsize=17)
plt.xlim(-5, 5)


# ## Joint analysis
# 
# In this section, we perform a joint analysis of the same data. Of course, joint fitting is considerably heavier than stacked one, and should always be handled with care. For brevity, we only show the analysis for a point source fitting without re-adding a diffuse component again. 
# 
# ### Data reduction

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Read the yaml file from disk\nconfig_joint = AnalysisConfig.read(path=path / "config_joint.yaml")\nanalysis_joint = Analysis(config_joint)\n\n# select observations:\nanalysis_joint.get_observations()\n\n# run data reduction\nanalysis_joint.get_datasets()')


# In[ ]:


# You can see there are 3 datasets now
print(analysis_joint.datasets)


# In[ ]:


# You can access each one by name or by index, eg:
print(analysis_joint.datasets[0])


# After the data reduction stage, it is nice to get a quick summary info on the datasets. 
# Here, we look at the statistics in the center of Map, by passing an appropriate `region`. To get info on the entire spatial map, omit the region argument.

# In[ ]:


analysis_joint.datasets.info_table()


# In[ ]:


models_joint = Models()

model_joint = model.copy(name="source-joint")
models_joint.append(model_joint)

for dataset in analysis_joint.datasets:
    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
    models_joint.append(bkg_model)

print(models_joint)


# In[ ]:


# and set the new model
analysis_joint.datasets.models = models_joint


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit_joint = Fit(analysis_joint.datasets)\nresult_joint = fit_joint.run()')


# ### Fit quality assessment and model residuals for a joint `Datasets` 

# We can access the results dictionary to see if the fit converged:

# In[ ]:


print(result_joint)


# Check best-fit parameters and error estimates:

# In[ ]:


print(models_joint)


# Since the joint dataset is made of multiple datasets, we can either:
# - Look at the residuals for each dataset separately. In this case, we can directly refer to the section `Fit quality and model residuals for a MapDataset` in this notebook
# - Look at a stacked residual map. 

# In the latter case, we need to properly stack the joint dataset before computing the residuals:

# In[ ]:


# TODO: clean this up

# We need to stack on the full geometry, so we use to geom from the stacked counts map.
stacked = MapDataset.from_geoms(**dataset_stacked.geoms)

for dataset in analysis_joint.datasets:
    # TODO: Apply mask_fit before stacking
    stacked.stack(dataset)

stacked.models = [model_joint]


# In[ ]:


stacked.plot_residuals_spatial(vmin=-1, vmax=1);


# Then, we can access the stacked model residuals as previously shown in the section `Fit quality and model residuals for a MapDataset` in this notebook.

# Finally, let us compare the spectral results from the stacked and joint fit:

# In[ ]:


def plot_spectrum(model, result, label, color):
    spec = model.spectral_model
    energy_range = [0.3, 10] * u.TeV
    spec.plot(
        energy_range=energy_range, energy_power=2, label=label, color=color
    )
    spec.plot_error(energy_range=energy_range, energy_power=2, color=color)


# In[ ]:


plot_spectrum(model, result, label="stacked", color="tab:blue")
plot_spectrum(model_joint, result_joint, label="joint", color="tab:orange")
plt.legend()


# ## Summary
# 
# Note that this notebook aims to show you the procedure of a 3D analysis using just a few observations. Results get much better for a more complete analysis considering the GPS dataset from the CTA First Data Challenge (DC-1) and also the CTA model for the Galactic diffuse emission, as shown in the next image:

# ![](images/DC1_3d.png)

# The complete tutorial notebook of this analysis is available to be downloaded in [GAMMAPY-EXTRA](https://github.com/gammapy/gammapy-extra) repository at https://github.com/gammapy/gammapy-extra/blob/master/analyses/cta_1dc_gc_3d.ipynb).

# ## Exercises
# 
# * Analyse the second source in the field of view: G0.9+0.1 and add it to the combined model.
# * Perform modeling in more details - Add diffuse component, get flux points.
