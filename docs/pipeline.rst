Pipeline Reference
==================

The mioXpektron pipeline provides end-to-end batch processing of ToF-SIMS
spectra. It chains recalibration, denoising, baseline correction,
normalization, peak detection, and cross-sample alignment into a single call.

Pipeline Steps
--------------

The pipeline executes these steps in order:

1. **Recalibration** (optional) --- convert channel numbers to m/z values
   using reference masses and a TOF model.
2. **Denoising** --- reduce noise with wavelet, Gaussian, median, or
   Savitzky-Golay filters.
3. **Baseline correction** --- remove broad background with AirPLS, AsLS, or
   other pybaselines methods.
4. **TIC normalization** --- scale spectra to a common Total Ion Current.
5. **Peak detection** --- find peaks using local maximum or CWT algorithms,
   with automatic noise estimation.
6. **Alignment** --- align detected peaks across samples by m/z tolerance,
   producing unified intensity and area matrices.

Configuration
-------------

.. code-block:: python

   from mioXpektron import PipelineConfig

   config = PipelineConfig(
       # Recalibration
       use_recalibration=True,
       reference_masses=[1.008, 22.99, 38.96, 58.07],
       output_folder_calibrated="calibrated_spectra",

       # Denoising
       denoise_method="wavelet",     # wavelet | gaussian | median | savitzky_golay | none
       denoise_params=None,          # dict of method-specific keyword arguments

       # Baseline
       baseline_method="airpls",
       baseline_params=None,
       clip_negative_after_baseline=True,

       # Normalization
       normalization_target=1e6,

       # Peak alignment
       mz_min=None,                  # optional m/z range filter
       mz_max=None,
       mz_tolerance=0.2,            # Da tolerance for cross-sample alignment
       mz_rounding_precision=1,

       # Parallelism
       max_workers=None,             # None = use all available cores

       # Adaptive parameterization (opt-in)
       auto_tune=False,             # True = derive mz_tolerance and normalization_target from data
   )

Running the Pipeline
--------------------

.. code-block:: python

   from mioXpektron import run_pipeline

   files = ["sample_01.txt", "sample_02.txt", "sample_03.txt"]

   intensity_df, area_df = run_pipeline(files, config=config)

With calibration data:

.. code-block:: python

   calib_channels = {
       "sample_01.txt": [100, 500, 1000, 2000],
       "sample_02.txt": [101, 502, 998, 2001],
   }

   intensity_df, area_df = run_pipeline(
       files,
       calib_channels_dict=calib_channels,
       config=config,
   )

Output Format
-------------

The pipeline returns two ``pandas.DataFrame`` objects:

**intensity_df**
   Rows = m/z values (aligned across samples), columns = sample names.
   Values are peak intensities after processing.

**area_df**
   Same structure as ``intensity_df`` but values are integrated peak areas.

Both DataFrames share the same m/z index, making them ready for downstream
statistical analysis.

Adaptive Parameterization
-------------------------

Set ``auto_tune=True`` to let the pipeline derive ``mz_tolerance`` and
``normalization_target`` from the data instead of using hardcoded defaults:

.. code-block:: python

   config = PipelineConfig(auto_tune=True)
   intensity_df, area_df = run_pipeline(files, config=config)

When ``auto_tune`` is active the pipeline:

1. Estimates ``mz_tolerance`` from median m/z spacing across a pilot sample.
2. Estimates ``normalization_target`` from median raw TIC across the batch.

All other parameters keep their defaults and can still be overridden manually.
See :doc:`modules/adaptive` for details on each estimator.

Reference Masses
^^^^^^^^^^^^^^^^

The pipeline now provides a canonical reference mass list
``DEFAULT_REFERENCE_MASSES`` (18 ions) used as the fallback when
``reference_masses`` is not provided. Import it directly:

.. code-block:: python

   from mioXpektron import DEFAULT_REFERENCE_MASSES

PipelineConfig Reference
------------------------

.. autoclass:: mioXpektron.PipelineConfig
   :members:
   :undoc-members:
