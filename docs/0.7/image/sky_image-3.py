import numpy as np
from gammapy.image import SkyImage
from astropy.coordinates import SkyCoord
from astropy.modeling.models import Gaussian2D
from astropy import units as u

BINSZ = 0.02
sigma = 0.2
ampl = 1. / (2 * np.pi * (sigma / BINSZ) ** 2)
sources = [Gaussian2D(ampl, 0, 0, sigma, sigma),
           Gaussian2D(ampl, 2, 0, sigma, sigma),
           Gaussian2D(ampl, 0, 2, sigma, sigma),
           Gaussian2D(ampl, 0, -2, sigma, sigma),
           Gaussian2D(ampl, -2, 0, sigma, sigma),
           Gaussian2D(ampl, 2, -2, sigma, sigma),
           Gaussian2D(ampl, -2, 2, sigma, sigma),
           Gaussian2D(ampl, -2, -2, sigma, sigma),
           Gaussian2D(ampl, 2, 2, sigma, sigma),]


image = SkyImage.empty(nxpix=201, nypix=201, binsz=BINSZ)
image.name = 'Flux'

for source in sources:
    # Evaluate on cut out
    pos = SkyCoord(source.x_mean, source.y_mean,
                   unit='deg', frame='galactic')
    cutout = image.cutout(pos, size=(3.2 * u.deg, 3.2 * u.deg))
    c = cutout.coordinates()
    l, b = c.galactic.l.wrap_at('180d'), c.galactic.b
    cutout.data = source(l.deg, b.deg)
    image.paste(cutout)

image.show()