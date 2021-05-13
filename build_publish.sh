#!/bin/bash

# create binary wheel distribution using a clean Python3 virtual environment
# will be placed in the /dist folder
python setup.py bdist_wheel --universal

# publish to PyPI using twine
# make sure to configure .pypirc file first
python -m twine upload --repository spart dist/*

# cleaning up. This part has to be adopted by Windows users
# the python calls above should be fine!
rm -rf .eggs/
rm -rf build/
rm -rf dist/

