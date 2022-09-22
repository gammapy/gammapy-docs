from gammapy.maps import Map
filename = "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts.fits.gz"
image = Map.read(filename)
image.smooth("0.1 deg").plot()