from gammapy.image import SkyImage

filename = '$GAMMAPY_EXTRA/datasets/fermi_2fhl/fermi_2fhl_gc.fits.gz'
counts = SkyImage.read(filename, hdu=2)
counts.name = 'Counts Smoothed'
counts.show()