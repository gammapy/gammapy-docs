# gammapy-docs

This repository is for the Gammapy documentation build and deploy on Github pages

It does **not** contain the sources for the Gammmapy documentation.
Those are in the docs` folder of the `gammapy` code repository.

## Overview

The `docs` folder of the `master` branch in this repo
gets served at https://gammapy.github.io/gammapy-docs/

It must contain the rendered HTML Gammapy Sphinx docs
in sub-folders, one for each version.

Example: http://gammapy.github.io/gammapy-docs/0.6

Special versions:

* `docs/dev` - the development version
* `docs/stable/index.html` - forwards to latest stable, e.g. `0.6`
* `docs/index.html` - forwards to `docs/dev`

## Howto


## Update dev version

```
mkdir build/dev
cd build/dev
git clone https://github.com/gammapy/gammapy.git # do this once
cd gammapy
git pull
time python setup.py build_docs
cp -r build/dev/gammapy/docs/_build/html/* docs/dev/
git add docs/dev
git commit -m 'update docs/dev'
git push
```

## Update a stable version

Same as before, except check out the right tag

```
git checkout v0.6
```

## Download from RTD

RTD doesn't support downloading the HTML page:
https://github.com/rtfd/readthedocs.org/issues/3242

However, one can use e.g. wget to do it
https://www.guyrutenberg.com/2014/05/02/make-offline-mirror-of-a-site-using-wget/

```

wget -mkEpnp http://docs.gammapy.org/en/v0.6/
mv docs.gammapy.org/en/v0.6/* docs/0.6
rm -r docs.gammapy.org
```

## TODO

* How to fetch the right version of `gammapy-extra`?
* How to set up a conda env for older versions?
* How to avoid the repo from growing forever, i.e. discarding old committed versions in `docs/dev`?


## Notes

* Gammapy v0.6 build doesn't work
