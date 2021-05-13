
# create binary wheel distribution
# will be placed in the /dist folder
python setup.py bdist_wheel --universal
rm -rf .eggs/
rm -rf build/

# publish to PyPI using twine
# make sure to configure .pypirc file first
python -m twine --pre upload repository spart dist/*

