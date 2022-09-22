import matplotlib.pyplot as plt
from gammapy.irf import EnergyDependentMultiGaussPSF
filename = '$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
psf.plot_containment(0.68, show_safe_energy=False)
plt.show()