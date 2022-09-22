from gammapy.maps import MapAxis, RegionNDMap
import numpy as np

# Create a RegionNDMap with 12 energy bins
energy_axis = MapAxis.from_bounds(100., 1e5, 12, interp='log', name='energy', unit='GeV')
region_map = RegionNDMap.create("icrs;circle(83.63, 22.01, 0.5)", axes=[energy_axis])

# Fill the data along the energy axis and plot
region_map.data = np.logspace(-2,3,12)
region_map.plot()