Quick Start
===========

This guide walks through a typical mioXpektron workflow: loading data,
processing it step by step, and extracting peaks.

Importing Data
--------------

mioXpektron reads tab- or comma-separated ToF-SIMS spectra with columns for
m/z values and intensities:

.. code-block:: python

   import mioXpektron as mx

   # Load a single spectrum
   mz, intensity, sample_name, group = mx.import_data("path/to/spectrum.txt")

The loader automatically detects separators, skips comment lines, and infers
sample names from the filename.

Step-by-Step Processing
-----------------------

Denoising
^^^^^^^^^

Reduce noise while preserving peak shapes:

.. code-block:: python

   denoised = mx.noise_filtering(intensity, method="wavelet")

Available methods: ``"wavelet"``, ``"gaussian"``, ``"median"``,
``"savitzky_golay"``, ``"none"``.

Baseline Correction
^^^^^^^^^^^^^^^^^^^

Remove broad background signals:

.. code-block:: python

   corrected = mx.baseline_correction(denoised, method="airpls")

Over 20 methods are available from the `pybaselines`_ library, including
``"airpls"``, ``"asls"``, ``"mor"``, ``"snip"``, and more.

.. _pybaselines: https://pybaselines.readthedocs.io/

Normalization
^^^^^^^^^^^^^

Normalize spectra using any of 14 available methods:

.. code-block:: python

   from mioXpektron import normalize

   # TIC normalization (default)
   normalized = normalize(corrected, method="tic", target_tic=1e6)

   # Poisson scaling (recommended before PCA)
   scaled = normalize(corrected, method="poisson")

   # Or use the direct function
   from mioXpektron import tic_normalization
   normalized = tic_normalization(corrected, target_tic=1e6)

Peak Detection
^^^^^^^^^^^^^^

Detect peaks with area integration:

.. code-block:: python

   peaks_df = mx.detect_peaks_with_area(corrected, snr_threshold=3.0)

For continuous wavelet transform (CWT) based detection:

.. code-block:: python

   peaks_df = mx.detect_peaks_cwt_with_area(corrected, min_snr=3.0)

Visualization
^^^^^^^^^^^^^

Plot spectra with annotated peaks:

.. code-block:: python

   mx.PlotPeak(corrected, peaks_df)

Automated Pipeline
------------------

For batch processing, use the built-in pipeline that chains all steps:

.. code-block:: python

   from mioXpektron import run_pipeline, PipelineConfig

   config = PipelineConfig(
       denoise_method="wavelet",
       baseline_method="airpls",
       normalization_target=1e6,
       mz_tolerance=0.2,
   )

   files = ["sample_01.txt", "sample_02.txt", "sample_03.txt"]
   intensity_df, area_df = run_pipeline(files, config=config)

The pipeline returns two DataFrames: an intensity matrix and an area matrix,
both aligned by m/z across all samples.

Mass Calibration
----------------

Calibrate channel-based spectra to m/z:

.. code-block:: python

   from mioXpektron import AutoCalibrator, AutoCalibConfig

   config = AutoCalibConfig(
       reference_masses=[12.0, 28.0, 56.0],
       model="quadratic",
   )

   calibrator = AutoCalibrator(config)
   calibrated_data = calibrator.calibrate(data)

For more control, use ``FlexibleCalibrator`` with explicit channel-to-mass
mappings. See :doc:`modules/calibration` for details.

Batch Processing
----------------

Process entire directories of spectra:

.. code-block:: python

   from mioXpektron import BatchDenoising, batch_tic_norm

   # Batch denoising
   denoiser = BatchDenoising(method="savgol", window_length=11)
   denoised_files = denoiser.process_directory("data/")

   # Batch normalization
   normalized = batch_tic_norm("data/", output_dir="normalized/")

Method Comparison
-----------------

Compare denoising strategies on your data:

.. code-block:: python

   from mioXpektron import compare_denoising_methods

   results = compare_denoising_methods(
       data,
       methods=["wavelet", "gaussian", "savgol"],
       metric="snr",
   )

Evaluate baseline correction approaches:

.. code-block:: python

   from mioXpektron import BaselineMethodEvaluator

   evaluator = BaselineMethodEvaluator()
   best_method = evaluator.evaluate(data)

Evaluate normalization strategies:

.. code-block:: python

   from mioXpektron import NormalizationEvaluator

   evaluator = NormalizationEvaluator(
       files=["spectra/*.txt"],
       methods=["tic", "median", "rms", "poisson", "sqrt", "vsn"],
   )
   results = evaluator.evaluate()
   evaluator.print_summary()
   evaluator.plot()

Next Steps
----------

- :doc:`pipeline` --- detailed pipeline configuration reference
- :doc:`modules/index` --- in-depth module documentation
- :doc:`modules/calibration` --- calibration models and strategies
- :doc:`modules/denoise` --- denoising algorithms and parameter tuning
