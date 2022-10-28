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

import inspect


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
]

autodoc_default_options = {
    "members": True,
    "member-order": "alphabetical",
    "imported-members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": False,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_static_path = []


autodoc_typehints_format = "short"
autodoc_class_signature = "separated"


def fix_pybind11_signatures(
    app, what, name, obj, options, signature, return_annotation
):
    def _remove_self(signature):
        arguments = signature[1:-1].split(", ")
        if arguments and arguments[0].startswith("self:"):
            arguments.pop(0)
        return "(%s)" % ", ".join(arguments)

    def _reformat_typehints(content):
        return content.replace(
            "thirdai._thirdai.",
            "thirdai." if autodoc_typehints_format == "fully-qualified" else "",
        )

    if signature is not None:
        signature = _remove_self(signature)
        signature = _reformat_typehints(signature)

    if return_annotation is not None:
        return_annotation = _reformat_typehints(return_annotation)

    return (signature, return_annotation)


def skip_pybind11_builtin_members(app, what, name, obj, skip, options):
    skipped_entries = {
        "__init__": ["self", "args", "kwargs"],
        "__new__": ["args", "kwargs"],
    }

    ref_arguments = skipped_entries.get(name)
    if ref_arguments is not None:
        try:
            arguments = list(inspect.signature(obj).parameters.keys())
            if arguments == ref_arguments:
                return True
        except ValueError:
            pass

    return None


def setup(app):
    app.add_css_file("custom.css")
    app.connect("autodoc-skip-member", skip_pybind11_builtin_members)
    app.connect("autodoc-process-signature", fix_pybind11_signatures)
