import numpy as np
from gammapy.maps import Map
from astropy.convolution import Ring2DKernel, Tophat2DKernel
from gammapy.detect import KernelBackgroundEstimator

counts = Map.create(npix=100, binsz=1)
counts.data += 42
counts.data[50][50] = 1000
source_kernel = Tophat2DKernel(3)
bkg_kernel = Ring2DKernel(radius_in=4, width=2)
kbe = KernelBackgroundEstimator(kernel_src=source_kernel.array,
                                kernel_bkg=bkg_kernel.array)
result = kbe.run({'counts':counts})
result['exclusion'].plot()