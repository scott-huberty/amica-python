# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AMICA-Python'
copyright = '2025, Scott Huberty'
author = 'Scott Huberty'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_gallery.gen_gallery',]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
html_static_path = ['_static']


sphinx_gallery_conf = {
    'examples_dirs': ['examples'],      # source examples
    'gallery_dirs': ['auto_examples'],  # generated gallery
    'filename_pattern': r'.*',          # include all example files
    'download_all_examples': False,
    'remove_config_comments': True,
}
