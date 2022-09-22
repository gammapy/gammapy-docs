from gammapy.irf import EDispKernelMap
from gammapy.maps import MapAxis
from astropy.coordinates import SkyCoord

# Create a test EDispKernelMap from a gaussian distribution
energy_axis_true = MapAxis.from_energy_bounds(1,10, 8, unit="TeV", name="energy_true")
energy_axis = MapAxis.from_energy_bounds(1,10, 5, unit="TeV", name="energy")

edisp_map = EDispKernelMap.from_gauss(energy_axis, energy_axis_true, 0.3, 0)
position = SkyCoord(ra=83, dec=22, unit='deg', frame='icrs')

edisp_kernel = edisp_map.get_edisp_kernel(position)

# We can quickly check the edisp kernel via the peek() method
edisp_kernel.peek()