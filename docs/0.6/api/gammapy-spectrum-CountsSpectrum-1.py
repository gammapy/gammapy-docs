from gammapy.spectrum import CountsSpectrum
import numpy as np
import astropy.units as u

ebounds = np.logspace(0,1,11) * u.TeV
data = np.arange(10) * u.ct
spec = CountsSpectrum(
    energy_lo=ebounds[:-1],
    energy_hi=ebounds[1:],
    data=data,
)
spec.plot(show_poisson_errors=True)