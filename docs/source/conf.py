# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os

import time_stream


project = "Time-Stream"
copyright = "2025, UKCEH"
author = "UKCEH"
release = time_stream.__version__
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    'jupyter_sphinx',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_design',
    'sphinx_tabs.tabs',
    'sphinx_contributors',
    'sphinx_iconify',
    'sphinxcontrib.mermaid',
    "sphinx_autodoc_typehints",
]

plot_formats = ['svg']
plot_include_source = False
plot_html_show_source_link = False
plot_html_show_formats = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for autodoc -----------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autoclass_content = "class"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "shibuya"
html_static_path = ["_static"]

# Custom CSS
html_css_files = [
    "css/custom.css",
]

html_context = {
    "license": "GNU GPL v3.0",
}

html_theme_options = {
    "accent_color": "blue",
    "github_url": "https://github.com/NERC-CEH/time-stream",
}

mermaid_version = "11.12.0"

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False

# -- Jupyter-sphinx settings -------------------------------------------------------
# Add examples path to Python's path
examples_path = os.path.abspath('../../src/time_stream/examples')
# Make sure jupyter-sphinx uses the same path
os.environ['PYTHONPATH'] = examples_path + os.pathsep + os.environ.get('PYTHONPATH', '')
