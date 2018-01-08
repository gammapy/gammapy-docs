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
