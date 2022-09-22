import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from gammapy.irf import EffectiveAreaTable, EnergyDispersion
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.spectrum import SpectrumEvaluator

e_true = np.logspace(-2, 2.5, 109) * u.TeV
e_reco = np.logspace(-2, 2, 73) * u.TeV

aeff = EffectiveAreaTable.from_parametrization(energy=e_true)
edisp = EnergyDispersion.from_gauss(e_true=e_true, e_reco=e_reco, sigma=0.3, bias=0)

model = PowerLawSpectralModel(index=2.3, amplitude="2.5e-12 cm-2 s-1 TeV-1", reference="1 TeV")

predictor = SpectrumEvaluator(model=model, aeff=aeff, edisp=edisp, livetime="1 hour")
predictor.compute_npred().plot_hist()
plt.show()