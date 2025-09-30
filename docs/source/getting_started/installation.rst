.. _installation:

============
Installation
============

.. rst-class:: lead

   Install **Time-Stream** as a Python package and verify your setup.

Requirements
============

- Python **3.12+**
- Recommended package manager: **pip** or `uv <https://docs.astral.sh/uv/getting-started/installation/>`_
- **Polars** (included in the dependencies of the Time-Stream package)

Install options
===============

.. tab-set::
    :class: outline padded-tabs

    .. tab-item:: :iconify:`material-icon-theme:uv` uv

        Follow `uv installation instructions <https://docs.astral.sh/uv/getting-started/installation/>`_ if you
        haven't already.

        .. code-block:: bash

            uv add git+https://github.com/NERC-CEH/time-stream.git@main

    .. tab-item:: :iconify:`devicon:pypi` pip

        .. code-block:: bash

            pip install git+https://github.com/NERC-CEH/time-stream.git@main

Importing
=========

To use the library, simply import into your Python script:

.. code-block:: python

   import time_stream as ts
