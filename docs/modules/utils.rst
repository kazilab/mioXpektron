Utilities
=========

The utils module provides file I/O, batch processing orchestration, and
statistical analysis tools.

Data Import
-----------

Load ToF-SIMS spectra from text files:

.. code-block:: python

   from mioXpektron import import_data

   mz, intensity, sample_name, group = import_data(
       "spectrum.txt",
       mz_min=1.0,
       mz_max=300.0,
   )

The importer:

- Auto-detects separators (tab, comma, space)
- Skips comment lines (``#``, ``//``)
- Infers sample names from filenames
- Infers sample groups from filename patterns
- Supports optional m/z range filtering

Batch Processing
----------------

Run parallel peak extraction and alignment across many spectra:

.. code-block:: python

   from mioXpektron.utils import batch_processing

   peaks_df, intensity_df, area_df = batch_processing(
       file_list,
       max_workers=4,
       mz_min=1.0,
       mz_max=300.0,
       normalization_target=1e6,
       mz_tolerance=0.2,
   )

Statistical Analysis
--------------------

The analysis submodule provides tools for downstream statistical analysis of
aligned peak matrices.

Benjamini-Hochberg FDR Correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mioXpektron.utils.analysis import bh_fdr

   q_values = bh_fdr(p_values)

Univariate Testing
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mioXpektron.utils.analysis import compute_univariate_tests

   results = compute_univariate_tests(intensity_df, groups)

Visualization Helpers
^^^^^^^^^^^^^^^^^^^^^

The analysis module includes plotting utilities for:

- Volcano plots
- PCA and UMAP projections
- ROC curves
- Heatmaps

API Reference
-------------

.. autofunction:: mioXpektron.utils.import_data
