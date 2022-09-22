import matplotlib.pyplot as plt
from gammapy.irf import EnergyDependentMultiGaussPSF
filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/irfs/psf.fits'
psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
psf.plot_containment(0.68, show_safe_energy=False)
plt.show()