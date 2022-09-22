from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from gammapy.image import SkyImage

filename = '$GAMMAPY_EXTRA/datasets/fermi_2fhl/fermi_2fhl_gc.fits.gz'
counts = SkyImage.read(filename, hdu=2)
position = SkyCoord(0, 0, frame='galactic', unit='deg')
size = Quantity([5, 5], 'deg')
cutout = counts.cutout(position, size)
cutout.show()