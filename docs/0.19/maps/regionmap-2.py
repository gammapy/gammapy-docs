from gammapy.maps import RegionGeom, Map
import numpy as np

m = Map.create(npix=100,binsz=3/100, skydir=(83.63, 22.01), frame='icrs')
m.data = np.add(*np.indices((100, 100)))

# A circle centered in the Crab position
circle = RegionGeom.create("icrs;circle(83.63, 22.01, 0.5)")

# A box centered in the same position
box = RegionGeom.create("icrs;box(83.63, 22.01, 1,2,45)")

# An ellipse in a different location
ellipse = RegionGeom.create("icrs;ellipse(84.63, 21.01, 0.3,0.6,-45)")

# An annulus in a different location
annulus = RegionGeom.create("icrs;annulus(82.8, 22.91, 0.1,0.3)")

m.plot(add_cbar=True)

# Default plotting settings
circle.plot_region()

# Different line styles, widths and colors
box.plot_region(lw=2, linestyle='--', ec='k')
ellipse.plot_region(lw=2, linestyle=':', ec='white')

# Filling the region with a color
annulus.plot_region(lw=2, ec='purple', fc='purple')