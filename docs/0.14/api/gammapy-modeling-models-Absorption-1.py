import matplotlib.pyplot as plt
import astropy.units as u
from gammapy.modeling.models import Absorption

# Load tables for z=0.5
redshift = 0.5
dominguez = Absorption.read_builtin('dominguez').table_model(redshift)
franceschini = Absorption.read_builtin('franceschini').table_model(redshift)
finke = Absorption.read_builtin('finke').table_model(redshift)

# start customised plot
energy_range = [0.08, 3] * u.TeV
ax = plt.gca()
opts = dict(energy_range=energy_range, energy_unit='TeV', ax=ax, flux_unit='')
franceschini.plot(label='Franceschini 2008', **opts)
finke.plot(label='Finke 2010', **opts)
dominguez.plot(label='Dominguez 2011', **opts)

# tune plot
ax.set_ylabel(r'Absorption coefficient [$\exp{(-\tau(E))}$]')
ax.set_xlim(energy_range.value)  # we get ride of units
ax.set_ylim([1.e-4, 2.])
ax.set_yscale('log')
ax.set_title('EBL models (z=' + str(redshift) + ')')
plt.grid(which='both')
plt.legend(loc='best') # legend

# show plot
plt.show()