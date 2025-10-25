# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Tell sphinx where to find my package
import os
import sys
sys.path.insert(0, os.path.abspath("../../src"))

project = 'AMICA-Python'
copyright = '2025, Scott Huberty'
author = 'Scott Huberty'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc", # For NumPy style docstrings
    'sphinx_gallery.gen_gallery', # For generating example galleries
    "sphinx.ext.autodoc", # For automatic API documentation generation
    "sphinx.ext.autosummary", # For generating summary tables
    "sphinx.ext.intersphinx", # For linking to other projects' docs
    "sphinx_design",  # e.g. for tabs
        ]

templates_path = ['_templates']
exclude_patterns = []


# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
html_static_path = ['_static']

# html theme options
html_theme_options = {
    "nav_links": [
        {"title": "About", "url": "about"},
        {"title": "Install", "url": "install"},
        {"title": "API", "url": "api/index"},
        {"title": "Examples", "url": "auto_examples/index"},
    ],
    "github_url": "https://github.com/scott-huberty/amica-python"
}

# Sphinx Gallery configuration ----------------------------------------------
sphinx_gallery_conf = {
    'examples_dirs': ['examples'],      # source examples
    'gallery_dirs': ['auto_examples'],  # generated gallery
    'filename_pattern': r'.*',          # include all example files
    'download_all_examples': False,
    'remove_config_comments': True,
    'backreferences_dir': 'gen_modules/backreferences',
    "doc_module": ("amica",),
}


# autosummary configuration ---------------------------------------------------
# Do NOT generate autosummary for the class page — we’ll rely on numpydoc
autosummary_generate = False
# autosummary_generate_overwrite = True

# NumPyDoc configuration -----------------------------------------------------
# Tell numpydoc to create a table of contents for class members
numpydoc_show_class_members = True # # show table of methods inline
numpydoc_class_members_toctree = False # do NOT generate a toctree/stubs for methods
numpydoc_show_inherited_class_members = False
# numpydoc_validate = True

# Tell numpydoc to create cross-references for parameter types
numpydoc_xref_aliases = {
    'BaseEstimator': 'sklearn.base.BaseEstimator',
    'TransformerMixin': 'sklearn.base.TransformerMixin',
    "FastICA": "sklearn.decomposition.FastICA",
}

# autodoc configuration -------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": False,
    "show-inheritance": True,
}