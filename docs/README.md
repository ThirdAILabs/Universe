# Building the docs

Through doxygen + breathe + exhale + sphinx, we generate docs for our C++
library and also the Python library.

```bash
python3 -m venv env #  Create a virtual environment
./env/bin/activate  #  Activate virtual environment

# Install required packages into virtual environment
python3 -m pip install -r requirements.txt


# Build and install thirdai (to avoid license issues).
# thirdai installed is required to generate module documentation.

# wheel is required to build binary distributions.
python3 -m pip install wheel 

# Build the ThirdAI wheel
# Use feature flags to expose all (remove licensing issues).
cd ../
THIRDAI_FEATURE_FLAGS=THIRDAI_EXPOSE_ALL \
    python3 setup.py bdist_wheel

# TODO(anyone): Simplify using requirements already in setup.py
python3 -m pip install -r requirements.txt
python3 -m pip install dist/*.whl

# Now that the package (thirdai) and requirements (for docs and thirdai) are
# installed, we can proceed to generate documentation.

cd docs 
make html

# By now, the documentation generated should be available in _build/html folder.
(cd _build/html && python3 -m http.server 8080)

# Navigate to localhost:8080 and you should be able to see the generated documentation.
```

# Building UML

Use vanilla doxygen with the supplied `Doxyfile.in`.

```bash
doxygen Doxyfile.in
(cd build/doc && python3 -m http.server 8080)
# Navigate to localhost:8080 on your browser.
```
