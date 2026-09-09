# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from pathlib import Path

import time_stream

# Make the documentation example modules importable as ``examples`` (e.g. ``from examples import aggregation``).
# They live in ``docs/source/examples`` - outside the packaged source tree - so they are not shipped,
# type-checked, or measured for coverage.
#   - ``sys.path``   covers in-process consumers (the ``plot`` directive).
#   - ``PYTHONPATH`` covers ``jupyter-execute``, which runs each block in a subprocess kernel that does not
#     inherit this process's ``sys.path``.
_DOCS_SOURCE = str(Path(__file__).parent)
sys.path.insert(0, _DOCS_SOURCE)
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [_DOCS_SOURCE, os.environ.get("PYTHONPATH", "")]))


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

# -- jupyter-sphinx --------------------------------------------------------------
# Kernels talk to the build over ZeroMQ. ipykernel warns on every start that a TCP transport is unencrypted, so
# use Unix domain sockets instead: no ports are opened, and the warning goes away. Windows has no "ipc"
# transport, so it keeps the default. Everything else here is jupyter-sphinx's own default, which must be
# repeated because setting this replaces the value rather than adding to it.
jupyter_execute_kwargs = {"timeout": -1, "allow_errors": True, "store_widget_state": True}

if sys.platform != "win32":
    from traitlets.config import Config

    jupyter_execute_kwargs["config"] = Config({"KernelManager": {"transport": "ipc"}})

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
    "nav_links": [
            {
                "title": "Getting started",
                "url": "getting_started/installation"
            },
            {
                "title": "User guide",
                "url": "user_guide/intro"
            },
            {
                "title": "Development",
                "url": "developer/contributing"
            },
            {
                "title": "API reference",
                "url": "api/time_frame"
            },
        ]
}

mermaid_version = "11.12.0"

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False
