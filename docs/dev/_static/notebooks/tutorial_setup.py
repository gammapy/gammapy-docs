
# coding: utf-8

# # Gammapy tutorial setup
# 
# Gammapy is a Python package, and you can install it in the usual way (with `conda` or `pip` or whatever package manager or distribution channel you use).
# 
# If you don't have something already, we recommend you use conda,
# because it works on Linux, Mac and Windows in the same way.
# 
# Caveat: Sherpa is not available on Windows.
# 
# This is just a short version of the commands to get set up using Anaconda.
# More detailed information is available in the [Gammapy installation instructions](http://docs.gammapy.org/en/latest/install.html).
# 
# ## Get Anaconda
# 
# Go to https://www.continuum.io/downloads and download Anaconda.
# 
# You can either use the conda command line tool to install software and manage
# environments, or use the `Navigator` app that comes with Anaconda if you prefer
# a graphical user interface.
# 
# There's some more information about conda, Anaconda and Miniconda [here](http://nbviewer.jupyter.org/github/jakevdp/WhirlwindTourOfPython/blob/master/00-Introduction.ipynb#Installation-with-conda)
# 
# ## Install a scientific Python stack
# 
# Install Python, Numpy, Scipy, IPython, Jupyter, Astropy and more via
# ```
# conda install python=3.5 anaconda
# ```
# 
# * We recommend that you use Python 3.5 at this time.
# * Everything will work identically if you prefer to use Python 2.7.
# * Python 3.6 just came out recently, so conda binary packages aren't available yet in all cases.
# 
# ## Install optional dependencies
# 
# Gammapy has some optional dependencies (e.g. Sherpa for modeling / fitting, reproject for one tutorial notebook to work with a radio image, ...).
# 
# Let's get those also:
# ```
# conda install -c conda-forge uncertainties
# conda install -c conda-forge regions reproject photutils
# conda install -c sherpa sherpa
# ```
# 
# If sometimes you find that some package isn't available for your system, e.g. if this fails:
# ```
# conda install -c conda-forge regions
# ```
# then you can always try to install it with pip:
# ```
# pip install regions
# ```
# The difference is that conda intalls binary packages, and pip installs source packages and compiles code if the package is partly written in C or Fortran.
# 
# ## Install Gammapy
# 
# Go to some location where you have ~ 1 GB of disk space:
# ```
# cd <scratch>
# ```
# 
# Install the development version of Gammapy:
# ```
# git clone https://github.com/gammapy/gammapy.git
# cd gammapy
# python -m pip install .
# ```
# 
# At the moment Gammapy is under heavy development, we're adding features and fix bugs every week. So for now, we really recommend you use the development version, i.e. the latest and greatest. We have many tests, so breaking existing functionality in the development version is rare.
# 
# ## Get gammapy-extra
# 
# Get the `gammapy-extra` repo with example datasets and the notebooks:
# ```
# cd ..
# git clone https://github.com/gammapy/gammapy-extra.git
# export GAMMAPY_EXTRA=$PWD/gammapy-extra
# ```
# 
# ## Start using Gammapy
# 
# To check if you're set up, try to import Gammapy and check which version you have.
# ```
# $ python
# Python 3.5.3 (default, Jan 21 2017, 15:44:58) 
# [GCC 4.2.1 Compatible Apple LLVM 8.0.0 (clang-800.0.42.1)] on darwin
# Type "help", "copyright", "credits" or "license" for more information.
# >>> import gammapy
# >>> gammapy
# <module 'gammapy' from '/Users/deil/Library/Python/3.5/lib/python/site-packages/gammapy/__init__.py'>
# >>> gammapy.__version__
# '0.6.dev4303'
# ```
# ```
# >>> 
# >>> significance(n_on=10, mu_bkg=4.2, method='lima')
# array([ 2.39791813])
# ```
# The "0.6.dev4303" means that it's the development version leading to 0.6 (i.e. the last stable release is 0.5 at this time).
# 
# Start the notebooks using one of the following commands (depends a bit on how you installed it):
# ```
# cd gammapy-extra/notebooks
# jupyter-notebook
# jupyter notebook
# ipython notebook
# ```
# 
# 
