
# coding: utf-8

# # CWT Algorithm

# This notebook tutorial shows how to work with `CWT` algorithm for detecting gamma-ray sources.
# 
# You can find the [docs here](http://docs.gammapy.org/0.8/api/gammapy.detect.CWT.html#gammapy.detect.CWT)
# and [source code on GitHub here](https://github.com/gammapy/gammapy/blob/master/gammapy/detect/cwt.py) for better understanding how the algorithm is constructed. 

# ## Setup
# 
# On this section we just import some packages that can be used (or maybe not) in this tutorial. You can also see the versions of the packages in the outputs below and notice that this notebook was written on Python 2.7. Don't worry about that because the code is also Python 3 compatible. 

# In[ ]:


# Render our plots inline
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15, 5)


# In[ ]:


import sys
import numpy as np
import scipy as sp

print("Python version: " + sys.version)
print("Numpy version: " + np.__version__)
print("Scipy version: " + sp.__version__)


# ## CWT Algorithm. PlayGround

# First of all we import the data which should be analysied.

# In[ ]:


import os
from astropy.io import fits
from astropy.coordinates import Angle, SkyCoord
from gammapy.maps import Map

filename = "$GAMMAPY_DATA/fermi_survey/all.fits.gz"

counts = Map.read(filename=filename, hdu="COUNTS")
background = Map.read(filename=filename, hdu="BACKGROUND")

width = Angle([20, 10], "deg")
position = counts.geom.center_skydir
counts = counts.cutout(position=position, width=width)
background = background.cutout(position=position, width=width)

data = dict(counts=counts, background=background)


# In[ ]:


fig = plt.figure(figsize=(15, 3))

ax = fig.add_subplot(121, projection=data["counts"].geom.wcs)
data["counts"].plot(vmax=10, ax=ax, fig=fig)

ax = fig.add_subplot(122, projection=data["background"].geom.wcs)
data["background"].plot(vmax=10, ax=ax, fig=fig)


# Let's explore how CWT works. At first define parameters of the algorithm.  An imperative parameter is kernels (`detect.CWTKernels` object). So we should create it.

# In[ ]:


# Input parameters for CWTKernels
N_SCALE = 2  # Number of scales considered.
MIN_SCALE = 6.  # First scale used.
STEP_SCALE = 1.3  # Base scaling factor.


# In[ ]:


from gammapy.detect import CWTKernels

cwt_kernels = CWTKernels(
    n_scale=N_SCALE, min_scale=MIN_SCALE, step_scale=STEP_SCALE
)
print(cwt_kernels.info_table)


# Other parameters are optional, in this demonstration define them all.

# In[ ]:


MAX_ITER = 10  # The maximum number of iterations of the CWT algorithm.
TOL = 1e-5  # Tolerance for stopping criterion.
SIGNIFICANCE_THRESHOLD = 2.  # Measure of statistical significance.
SIGNIFICANCE_ISLAND_THRESHOLD = (
    None
)  # Measure is used for cleaning of isolated pixel islands.
REMOVE_ISOLATED = True  # If True, isolated pixels will be removed.
KEEP_HISTORY = True  # If you want to save images of all the iterations


# Let's start to analyse input data. Import Logging module to see how the algorithm works during data analysis.

# In[ ]:


from gammapy.detect import CWT
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
cwt = CWT(
    kernels=cwt_kernels,
    tol=TOL,
    significance_threshold=SIGNIFICANCE_THRESHOLD,
    significance_island_threshold=SIGNIFICANCE_ISLAND_THRESHOLD,
    remove_isolated=REMOVE_ISOLATED,
    keep_history=KEEP_HISTORY,
)


# In order to the algorithm was able to analyze source images, you need to convert them to a special format, i.e. create an CWTData object. Do this.

# In[ ]:


from gammapy.detect import CWTKernels, CWTData

cwt_data = CWTData(
    counts=data["counts"], background=data["background"], n_scale=N_SCALE
)


# In[ ]:


# Start the algorithm
cwt.analyze(cwt_data)


# ## Results of analysis
# 
# Look at the results of CWT algorithm. Print all the images.

# In[ ]:


PLOT_VALUE_MAX = 5
FIG_SIZE = (15, 35)

fig = plt.figure(figsize=FIG_SIZE)
images = cwt_data.images()
for index, (name, image) in enumerate(images.items()):
    ax = fig.add_subplot(len(images), 2, index + 1, projection=image.geom.wcs)
    image.plot(vmax=PLOT_VALUE_MAX, fig=fig, ax=ax)
    plt.title(name)  # Maybe add a Name in SkyImage.plots?


# As you can see in the implementation of CWT above, it has the parameter `keep_history`. If you set to it `True`-value, it means that CWT would save all the images from iterations. Algorithm keeps images of only last CWT start.  Let's do this in the demonstration.

# In[ ]:


history = cwt.history
print(
    "Number of iterations: {0}".format(len(history) - 1)
)  # -1 because CWT save start images too


# Let's have a look, what's happening with images after the first iteration.

# In[ ]:


N_ITER = 1
assert 0 < N_ITER < len(history)
data_iter = history[N_ITER]

fig = plt.figure(figsize=FIG_SIZE)
images_iter = data_iter.images()
for index, (name, image) in enumerate(images_iter.items()):
    ax = fig.add_subplot(
        len(images_iter), 2, index + 1, projection=image.geom.wcs
    )
    image.plot(vmax=PLOT_VALUE_MAX, fig=fig, ax=ax)
    plt.title(name)  # Maybe add a Name in SkyImage.plots?


# You can get the information about the one particular image in that way: 

# In[ ]:


print(data_iter.image_info(name="approx_bkg"))


# You can also get the information about cubes. Or information about all the data. 

# In[ ]:


print(data_iter.cube_info(name="support", per_scale=True))


# In[ ]:


print(data_iter.cube_info(name="support", per_scale=False))


# Also you can see the difference betwen the iterations in that way:

# In[ ]:


history = cwt.history  # get list of 'CWTData' objects
difference = (
    history[1] - history[0]
)  # get new `CWTData` obj, let's work with them


# In[ ]:


print(difference.cube_info("support"))


# In[ ]:


difference.info_table.show_in_notebook()


# In[ ]:


fig = plt.figure(figsize=FIG_SIZE)
images_diff = difference.images()
for index, (name, image) in enumerate(images_diff.items()):
    ax = fig.add_subplot(
        len(images_diff), 2, index + 1, projection=image.geom.wcs
    )
    image.plot(vmax=PLOT_VALUE_MAX, fig=fig, ax=ax)
    plt.title(name)  # Maybe add a Name in SkyImage.plots?


# You can save the results if you want

# In[ ]:


# cwt_data.write('test-cwt.fits', True)

