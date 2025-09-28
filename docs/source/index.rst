.. _index:

===========
Time-Stream
===========

.. rst-class:: lead

    **Time S**\eries **T**\oolkit for **R**\apid **E**\nvironmental **A**\nalysis and **M**\onitoring: An open-source
    Python library for handling and analysing timeseries data with a focus on maintaining the integrity of the
    temporal properties of the data.

.. container:: buttons

    `Docs </user_guide/>`_
    `GitHub <https://github.com/NERC-CEH/time-stream>`_

.. grid:: 1 1 2 3
    :gutter: 2
    :padding: 0
    :class-row: surface

    .. grid-item-card:: :octicon:`history` TimeSeries Data Model

        A powerful core object for managing time series with integrated metadata and flags.

    .. grid-item-card:: :octicon:`calendar` Time Periods & Alignments

        In-built functionality for handling timeseries resolution, periodicity and alignment.

    .. grid-item-card:: :octicon:`stack` Aggregation

        Robust time-based aggregation methods which also provide rich supplementary data about data completeness.

    .. grid-item-card:: :octicon:`report` Flag Systems & QC

        Define bitwise flag systems, manage flag columns, and run quality checks.

    .. grid-item-card:: :octicon:`pivot-column` Infilling

        Fill gaps in your timeseries with interpolation and advanced infill strategies, safely and reproducibly.

    .. grid-item-card:: :octicon:`info` Metadata

        Keep important metadata about your timeseries along with the data. Either timeseries-level metadata,
        or metadata of individual columns.

Why Time-Stream?
================

The goal of Time-Stream is to provide a user friendly Python library for processing time series data, particularly
in the hydrological and environmental domain. It is built on top of `Polars <https://docs.pola.rs/>`_, which handles
efficient DataFrame processes, whilst adding on specific functionality to help you manage time properties such as
resolution, periodicity, and anchor points.

- **Explicit time property management**: Perform methods on your data without worrying about whether it's handling your time data correctly.
- **Domain knowledge**: Built by software engineers and data scientists from `UKCEH <https://www.ceh.ac.uk/>`_, with years of experience working with hydrological and environmental data.
- **Building blocks**: Modular design for aggregation, flagging, QC, and infilling.
- **Polars performance**: Polars under the hood, vectorized paths where possible.

.. container:: image-row

   .. container:: image-item

      .. figure:: _static/ukceh_logo.png
         :alt: UKCEH
         :height: 100px
         :target: https://www.ceh.ac.uk

   .. container:: image-item

      .. figure:: _static/fdri_logo.png
         :alt: FDRI
         :height: 100px
         :target: https://fdri.org.uk

Community
=========

Developed at `UKCEH <https://www.ceh.ac.uk/>`_, welcoming community engagement and contributions.
Below are some of the top contributors to the project:

.. contributors:: NERC-CEH/time-stream
    :avatars:

Contributing
============

We welcome contributions! See :doc:`developer/contributing` for setup, coding standards, documentation
guidelines, and CI.

License
=======

This project is licensed under the `GNU General Public License v3.0 <https://github.com/NERC-CEH/time-stream/blob/main/LICENSE>`_.

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Getting started

    getting_started/installation
    getting_started/quick-start
    getting_started/concepts

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: User guide

    user_guide/intro
    user_guide/aggregation
    user_guide/infilling
    user_guide/quality_control


.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Developer guide

    developer/contributing
    developer/documentation
