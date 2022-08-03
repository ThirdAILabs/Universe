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
# Alphabetized by last name
author = " Josh Engels, Benito Geordie, Vihan Lakshman, Tharun Medini, Nicholas Meisburger, Anshumali Shrivastava, David Torres, Patrick Yan, Henry Zhang"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "breathe",
    "exhale",
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinxarg.ext",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", 
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
# html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------

breathe_projects = {"thirdai": "./doxygen/xml"}
breathe_default_project = "thirdai"

doxygen_config = """
INPUT                = ../bolt ../hashtable ../hashing ../search ../datasets ../exceptions
EXCLUDE             += ../deps
EXCLUDE             += env
EXCLUDE_PATTERNS     = *.md *.txt **/tests/** **/python_bindings/** **/python_tests/**
FILE_PATTERNS       += *.cu
EXTENSION_MAPPING   += cu=C++ inc=C++
ENABLE_PREPROCESSING = YES
JAVADOC_AUTOBRIEF    = YES
WARN_IF_UNDOCUMENTED = NO
"""

exhale_args = {
    "containmentFolder": "./api",
    "rootFileName": "library_index.rst",
    "rootFileTitle": "Library API",
    "doxygenStripFromPath": "..",
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": doxygen_config.strip(),
}

primary_domain = "cpp"
highlight_language = "cpp"

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

