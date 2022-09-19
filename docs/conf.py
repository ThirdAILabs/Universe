import textwrap

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "thirdai"
copyright = "2022, t"
author = "ThirdAI Engineering"


# -- General configuration ---------------------------------------------------

autodoc_mock_imports = ["ray", "toml", "torch", "transformers"]

extensions = [
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.imgmath",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "breathe",
    "exhale",
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinxarg.ext",
]

autodoc_default_options = {
    "members": True,
    "member-order": "alphabetical",
    "imported-members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "build",
    "doxygen",
    "venv",
    "README.md",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_static_path = []


# -- Extension configuration -------------------------------------------------

breathe_projects = {"thirdai": "./doxygen/xml"}
breathe_default_project = "thirdai"

doxygen_config = """
INPUT                = ../bolt \
                       ../bolt_vector \
                       ../dataset \
                       ../compression \
                       ../hashing \
                       ../hashtable \
                       ../licensing \
                       ../search \
                       ../utils

EXCLUDE_PATTERNS    += */deps/*
EXCLUDE_PATTERNS    += */tests/* 
EXCLUDE_PATTERNS    += */python_bindings/* 
EXCLUDE_PATTERNS    += */python_tests/*
EXCLUDE_PATTERNS    += *.md *.txt 
FILE_PATTERNS       += *.cu
EXTENSION_MAPPING   += cu=C++ inc=C++
ENABLE_PREPROCESSING = YES
JAVADOC_AUTOBRIEF    = YES
WARN_IF_UNDOCUMENTED = YES
"""

exhale_args = {
    "containmentFolder": "./api",
    "doxygenStripFromPath": "..",
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": doxygen_config.strip(),
    "rootFileName": "library_index.rst",
    "rootFileTitle": "thirdai (C++)",
    "afterTitleDescription": textwrap.dedent(
        """
       .. note::

       The following documentation presents the C++ API.  The Python API
       generally mirrors the C++ API, but some methods may not be available in
       Python or may perform different actions.
    """
    ),
}

primary_domain = "cpp"
highlight_language = "cpp"

# A trick to include

# A trick to include markdown files from outside the source directory using
# 'mdinclude'. Warning: all other markdown files not included via 'mdinclude'
# will be rendered using recommonmark as recommended by Sphinx
from m2r import MdInclude


def setup(app):
    # from m2r to make `mdinclude` work
    app.add_config_value("no_underscore_emphasis", False, "env")
    app.add_config_value("m2r_parse_relative_links", False, "env")
    app.add_config_value("m2r_anonymous_references", False, "env")
    app.add_config_value("m2r_disable_inline_math", False, "env")
    app.add_directive("mdinclude", MdInclude)
