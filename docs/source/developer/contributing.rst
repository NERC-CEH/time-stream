Contributing
=============

Writing documentation
---------------------

This section explains how to write documentation for the Time-Stream Package using Sphinx.
Comprehensive documentation is crucial for helping users understand and effectively use the package.

Documentation Structure
^^^^^^^^^^^^^^^^^^^^^^^

Our documentation is organized as follows:

.. code-block:: text

    docs/
    ├── source/
    │   ├── _static/           # Static assets (CSS, images)
    │   ├── getting_started/   # Installation and basic usage
    │   ├── user_guide/        # In-depth guides for features
    │   ├── api/               # API reference documentation
    │   ├── developer/         # Developer guides (like this one)
    │   ├── conf.py            # Sphinx configuration
    │   └── index.rst          # Main index page
    ├── Makefile              # Build commands for Unix
    └── make.bat              # Build commands for Windows

Setting Up for Documentation
^^^^^^^^^^^^^^^^^^^^^^^

Before writing documentation:

1. Install the required docs packages:

   .. code-block:: bash

       pip install .[docs]

2. Familiarize yourself with reStructuredText (RST) syntax:

   - Headings use underlines: ``=====`` for main headings, ``-----`` for subheadings, etc.
   - Lists start with ``*``, ``-``, or numbers ``1.``, ``2.``, etc.
   - Links use ```Text <URL>`_`` or ``:doc:`path/to/doc``` for internal links
   - See https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html for more information

Creating a New Documentation Page
^^^^^^^^^^^^^^^^^^^^^^^

To add a new page to the documentation:

1. Create a new ``.rst`` file in the appropriate directory
2. Start with a title and introduction
3. Add the page to the relevant toctree in ``index.rst`` or a section index

Including Code Examples
^^^^^^^^^^^^^^^^^^^^^^^

Good documentation includes clear code examples. The way we are including code snippets in these documentation is
to write Python within actual script files, saved within the ``src/time_stream/examples`` directory. The code can
then be included in the documentation using the ``literalinclude`` block, and executed using the ``jupyter-execute``
block.

Code in the example Python file should be organised into individual functions, and use "start-after" and "end-before"
markers. This ensures that the ``literalinclude`` and ``jupyter-execute`` blocks know which bit of code to show/execute.

This approach:
1. Shows the code exactly as it appears in your example file
2. Executes the code and displays its output
3. Keeps example code in maintainable, testable Python files
4. Ensures documentation examples are accurate and up-to-date

Using start-after / end-before Markers
"""""""""""""""""""""""

To include specific sections from a file, add marker comments to your code:

.. code-block:: python

    # example.py
    import time_series

    def example_function():
        # [start_block_1]
        # Create a TimeSeries
        dates = [datetime(2023, 1, i) for i in range(1, 5)]
        values = [10, 12, 15, 14]

        df = pl.DataFrame({
            "timestamp": dates,
            "temperature": values
        })

        ts = TimeSeries(df=df, time_name="timestamp")
        # [end_block_1]

Then in your RST file:

.. code-block:: rst

    .. literalinclude:: ../../../src/time_series/examples/example.py
       :language: python
       :start-after: [start_block_1]
       :end-before: [end_block_1]
       :dedent:

Key options for ``literalinclude``:

- ``:language:``: Syntax highlighting language
- ``:start-after:``: Start including after a specific string
- ``:end-before:``: Stop including before a specific string
- ``:dedent:``: Remove indented spaces from each line to make the code snippet in the documentation flush


Executing Code with jupyter-execute
"""""""""""""""""""""""

To show the output of the code snippet, use ``jupyter-execute`` and call the function containing the code snippet:

.. code-block:: rst

    .. jupyter-execute::
       :hide-code:
       import examples
       ts = examples.example_function()

Key options for ``jupyter-execute``:

- ``:hide-code:``: Show only the output, not the code

Building Documentation
^^^^^^^^^^^^^^^^^^^^^^^

To build the documentation:

.. code-block:: bash

    cd docs
    make html

View the result by opening ``docs/_build/html/index.html`` in a browser.

Review the build output for warnings and errors.

Example Documentation Workflow
^^^^^^^^^^^^^^^^^^^^^^^

1. **Write example code**: Create a Python file in ``src/time_series/examples``
2. **Test the example**: Ensure it works correctly
3. **Add marker comments**: Add ``[start_block_X]`` and ``[end_block_X]`` markers
4. **Create documentation**: Write an RST file referencing the example
5. **Build and verify**: Build the documentation and check the results
6. **Review and refine**: Ensure clarity and completeness
