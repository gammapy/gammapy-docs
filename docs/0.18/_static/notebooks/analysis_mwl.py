#!/usr/bin/env python
# coding: utf-8

# # Joint modeling, fitting, and serialization
# 

# ## Prerequisites
# 
# - Handling of Fermi-LAT data with gammapy [see the corresponding tutorial](fermi_lat.ipynb)
# - Knowledge of spectral analysis to produce 1D On-Off datasets, [see the following tutorial](spectrum_analysis.ipynb)
# - Using flux points to directly fit a model (without forward-folding)  [see the SED fitting tutorial](sed_fitting.ipynb)
# 
# ## Context
# 
# Some science studies require to combine heterogeneous data from various instruments to extract physical informations. In particular, it is often useful to add flux measurements of a source at different energies to an analysis to better constrain the wide-band spectral parameters. This can be done using a joint fit of heterogeneous datasets.
#  
# **Objectives: Constrain the spectral parameters of the gamma-ray emission from the Crab nebula between 10 GeV and 100 TeV, using a 3D Fermi dataset, a H.E.S.S. reduced spectrum and HAWC flux points.**
# 
# ## Proposed approach
# 
# This tutorial illustrates how to perfom a joint modeling and fitting of the Crab Nebula spectrum using different datasets.
# The spectral parameters are optimized by combining a 3D analysis of Fermi-LAT data, a ON/OFF spectral analysis of HESS data, and flux points from HAWC.
# 
# In this tutorial we are going to use pre-made datasets. We prepared maps of the Crab region as seen by Fermi-LAT using the same event selection than the [3FHL catalog](https://arxiv.org/abs/1702.00664) (7 years of data with energy from 10 GeV to 2 TeV). For the HESS ON/OFF analysis we used two observations from the [first public data release](https://arxiv.org/abs/1810.04516) with a significant signal from energy of about 600 GeV to 10 TeV. These observations have an offset of 0.5° and a zenith angle of 45-48°. The HAWC flux points data are taken from a [recent analysis](https://arxiv.org/pdf/1905.12518.pdf) based on 2.5 years of data with energy between 300 Gev and 300 TeV. 
# 
# ## The setup
# 

# In[ ]:


from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling import Fit
from gammapy.modeling.models import Models
from gammapy.datasets import Datasets, FluxPointsDataset, SpectrumDatasetOnOff
from gammapy.estimators import FluxPoints, FluxPointsEstimator
from gammapy.maps import MapAxis
from pathlib import Path


# ## Data and models files
# 
# 
# The datasets serialization produce YAML files listing the datasets and models. In the following cells we show an example containning only the Fermi-LAT dataset and the Crab model. 
# 
# Fermi-LAT-3FHL_datasets.yaml:

# In[ ]:


get_ipython().system('cat $GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml')


# We used as model a point source with a log-parabola spectrum. The initial parameters were taken from the latest Fermi-LAT catalog [4FGL](https://arxiv.org/abs/1902.10045), then we have re-optimized the spectral parameters for our dataset in the 10 GeV - 2 TeV energy range (fixing the source position).
# 
# Fermi-LAT-3FHL_models.yaml:

# In[ ]:


get_ipython().system('cat $GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml')


# ## Reading  different datasets
# 
# 
# ### Fermi-LAT 3FHL: map dataset for 3D analysis
# For now we let's use the datasets serialization only to read the 3D `MapDataset` associated to Fermi-LAT 3FHL data and models.

# In[ ]:


path = Path("$GAMMAPY_DATA/fermi-3fhl-crab")
filename = path / "Fermi-LAT-3FHL_datasets.yaml"

datasets = Datasets.read(filename=filename)


# In[ ]:


models = Models.read(path / "Fermi-LAT-3FHL_models.yaml")
print(models)


# We get the Crab model in order to share it with the other datasets

# In[ ]:


print(models["Crab Nebula"])


# ### HESS-DL3: 1D ON/OFF dataset for spectral fitting
# 
# The ON/OFF datasets can be read from PHA files following the [OGIP standards](https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node5.html).
# We read the PHA files from each observation, and compute a stacked dataset for simplicity.
# Then the Crab spectral model previously defined is added to the dataset.

# In[ ]:


datasets_hess = Datasets()

for obs_id in [23523, 23526]:
    dataset = SpectrumDatasetOnOff.from_ogip_files(
        f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obs_id}.fits"
    )
    datasets_hess.append(dataset)

dataset_hess = datasets_hess.stack_reduce(name="HESS")

datasets.append(dataset_hess)


# In[ ]:


print(datasets)


# ### HAWC: 1D dataset for flux point fitting
# 
# The HAWC flux point are taken from https://arxiv.org/pdf/1905.12518.pdf. Then these flux points are read from a pre-made FITS file and passed to a `FluxPointsDataset` together with the source spectral model.
# 

# In[ ]:


# read flux points from https://arxiv.org/pdf/1905.12518.pdf
filename = "$GAMMAPY_DATA/hawc_crab/HAWC19_flux_points.fits"
flux_points_hawc = FluxPoints.read(filename)
dataset_hawc = FluxPointsDataset(data=flux_points_hawc, name="HAWC")

datasets.append(dataset_hawc)


# In[ ]:


print(datasets)


# ## Datasets serialization
# 
# The `datasets` object contains each dataset previously defined. 
# It can be saved on disk as datasets.yaml, models.yaml, and several data files specific to each dataset. Then the `datasets` can be rebuild later from these files.

# In[ ]:


path = Path("crab-3datasets")
path.mkdir(exist_ok=True)

filename = path / "crab_10GeV_100TeV_datasets.yaml"

datasets.write(filename, overwrite=True)


# In[ ]:


get_ipython().system('cat crab-3datasets/crab_10GeV_100TeV_datasets.yaml')


# In[ ]:


datasets = Datasets.read(filename)
datasets.models = models


# In[ ]:


print(datasets)


# ## Joint analysis
# 
# We run the fit on the `Datasets` object that include a dataset for each instrument
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit_joint = Fit(datasets)\nresults_joint = fit_joint.run()\nprint(results_joint)')


# Let's display only the parameters of the Crab spectral model

# In[ ]:


crab_spec = datasets[0].models["Crab Nebula"].spectral_model
print(crab_spec)


# We can compute flux points for Fermi-LAT and HESS datasets in order plot them together with the HAWC flux point.

# In[ ]:


# compute Fermi-LAT and HESS flux points
energy_edges = MapAxis.from_energy_bounds("10 GeV", "2 TeV", nbin=5).edges

flux_points_fermi = FluxPointsEstimator(
    energy_edges=energy_edges, source="Crab Nebula"
).run([datasets["Fermi-LAT"]])


energy_edges = MapAxis.from_bounds(
    1, 15, nbin=6, interp="log", unit="TeV"
).edges
flux_points_hess = FluxPointsEstimator(
    energy_edges=energy_edges, source="Crab Nebula"
).run([datasets["HESS"]])


# Now, Let's plot the Crab spectrum fitted and the flux points of each instrument.
# 

# In[ ]:


# display spectrum and flux points
energy_range = [0.01, 120] * u.TeV
plt.figure(figsize=(8, 6))
ax = crab_spec.plot(energy_range=energy_range, energy_power=2, label="Model")
crab_spec.plot_error(ax=ax, energy_range=energy_range, energy_power=2)
flux_points_fermi.plot(ax=ax, energy_power=2, label="Fermi-LAT")
flux_points_hess.plot(ax=ax, energy_power=2, label="HESS")
flux_points_hawc.plot(ax=ax, energy_power=2, label="HAWC")
plt.legend();


# In[ ]:




