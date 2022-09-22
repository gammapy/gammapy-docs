import naima
from gammapy.spectrum.models import NaimaModel
import astropy.units as u
import matplotlib.pyplot as plt

particle_distribution = naima.models.ExponentialCutoffPowerLaw(1e30 / u.eV, 10 * u.TeV, 3.0, 30 * u.TeV)
radiative_model = naima.radiative.InverseCompton(
    particle_distribution,
    seed_photon_fields=[
        "CMB",
        ["FIR", 26.5 * u.K, 0.415 * u.eV / u.cm ** 3],
    ],
    Eemin=100 * u.GeV,
)

model = NaimaModel(radiative_model, distance=1.5 * u.kpc)

opts = {
    "energy_range" : [10 * u.GeV, 80 * u.TeV],
    "energy_power" : 2,
    "flux_unit" : "erg-1 cm-2 s-1",
}

# Plot the total inverse Compton emission
model.plot(label='IC (total)', **opts)

# Plot the separate contributions from each seed photon field
for seed, ls in zip(['CMB','FIR'], ['-','--']):
    model = NaimaModel(radiative_model, seed=seed, distance=1.5 * u.kpc)
    model.plot(label="IC ({})".format(seed), ls=ls, color="gray", **opts)

plt.legend(loc='best')
plt.show()