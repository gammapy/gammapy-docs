from gammapy.image import SkyImage
filename = '$GAMMAPY_EXTRA/datasets/fermi_2fhl/fermi_2fhl_vela.fits.gz'
image = SkyImage.read(filename, hdu=2)
image.show()