.. _time_frame_api:

=========
TimeFrame
=========

.. currentmodule:: time_stream

.. autoclass:: TimeFrame

Attributes
==========

.. autosummary::
    :nosignatures:
    :toctree: _api/

    ~TimeFrame.df
    ~TimeFrame.resolution
    ~TimeFrame.offset
    ~TimeFrame.alignment
    ~TimeFrame.periodicity
    ~TimeFrame.time_anchor
    ~TimeFrame.time_name
    ~TimeFrame.columns
    ~TimeFrame.flag_columns
    ~TimeFrame.data_columns
    ~TimeFrame.metadata
    ~TimeFrame.column_metadata

Methods
=======

Builders
--------

.. autosummary::
    :nosignatures:
    :toctree: _api/

    ~TimeFrame.with_df
    ~TimeFrame.with_periodicity
    ~TimeFrame.with_metadata
    ~TimeFrame.with_column_metadata
    ~TimeFrame.with_flag_system

General
-------

.. autosummary::
    :nosignatures:
    :toctree: _api/

    ~TimeFrame.sort_time
    ~TimeFrame.pad
    ~TimeFrame.select

Operations
----------

.. autosummary::
    :nosignatures:
    :toctree: _api/

    ~TimeFrame.aggregate
    ~TimeFrame.infill
    ~TimeFrame.qc_check
    ~TimeFrame.calculate_min_max_envelope

Flagging
--------

.. autosummary::
    :nosignatures:
    :toctree: _api/

    ~TimeFrame.register_flag_system
    ~TimeFrame.get_flag_system
    ~TimeFrame.register_flag_column
    ~TimeFrame.init_flag_column
    ~TimeFrame.get_flag_column
    ~TimeFrame.add_flag
    ~TimeFrame.remove_flag
