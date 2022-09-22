# gammapy-docs

This repository is for the Gammapy documentation build and deploy on Github pages

It does **not** contain the sources for the Gammmapy documentation.
Those are in the `docs` folder of the `gammapy` code repository.

## Overview

The `docs` folder of the `main` branch in this repo
gets served at https://gammapy.github.io/gammapy-docs/

It must contain the rendered HTML Gammapy Sphinx docs
in sub-folders, one for each version.

Example: http://gammapy.github.io/gammapy-docs/0.6

Special versions:

* `docs/dev` - the development version
* `docs/stable/index.html` - forwards to latest stable, e.g. `0.6`
* `docs/index.html` - forwards to `docs/stable`

## Howto

First make sure you have the full datasets available under $GAMMAPY_DATA. They are downloaded with `gammapy download datasets`.

### Update dev version

*Beware: the command `git clean -fdx` will remove all untracked files and directories* 

Below we assume that the gammapy and the gammapy-docs are installed in the same directory.

```
cd gammapy
git clean -fdx
git pull
pip install -e .
time make docs-sphinx
cd ../gammapy-docs
rm -r docs/dev
cp -r ../gammapy/docs/_build/html docs/dev
git add docs/dev
git commit -m 'update docs/dev'
git push
```

### Update a stable version

```
cd build
mkdir 0.10  # or whatever the version is
cd 0.10
git clone https://github.com/gammapy/gammapy.git
cd gammapy
git checkout v0.10
pip install -e .
time make docs-sphinx release=v0.10
cd ../../..
cp -r build/0.10/gammapy/docs/_build/html docs/0.10
git add docs/0.10
git commit -m 'Add docs/0.10'
git push
```

Then update `stable/index.html` to point to the new stable version.

## Very old versions

An archive of very old versions of built Gammapy docs is available here:
https://github.com/cdeil/gammapy-docs-rtd-archive

## TODO

* How to set up a conda env for older versions?
* How to avoid the repo from growing forever, i.e. discarding old committed versions in `docs/dev`?

## Notes

* Gammapy v0.6 build doesn't work
