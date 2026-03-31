Adaptive Parameterization
=========================

The ``adaptive`` module provides data-driven estimators that replace
fixed thresholds and constants with values derived from your spectra.
All estimators are **opt-in**: existing defaults remain unchanged unless
you explicitly activate adaptive tuning.

Motivation
----------

Many pipeline parameters (calibration tolerance, outlier threshold,
normalization target, denoise cutoffs, baseline quantiles) are typically
set by convention. The adaptive module estimates these values from a
pilot sample of your spectra, making the pipeline more robust across
instruments, sample types, and acquisition conditions.

Quick Start
-----------

The simplest way to activate adaptive parameterization is to set
``auto_tune=True`` on the relevant config:

.. code-block:: python

   from mioXpektron import FlexibleCalibrator, FlexibleCalibConfig

   config = FlexibleCalibConfig(
       reference_masses=my_masses,
       calibration_method="quad_sqrt",
       auto_tune=True,           # <-- activates data-driven estimation
   )

   calibrator = FlexibleCalibrator(config)
   summary = calibrator.calibrate(file_list)

When ``auto_tune=True``, the calibrator will:

1. Estimate ``autodetect_tol_da`` from peak widths near calibrant masses.
2. Derive ``multisegment_breakpoints`` from the calibrant distribution.
3. Enable ``auto_screen_reference_masses`` automatically.
4. After the first fit pass, derive ``outlier_threshold`` from residuals.
5. Derive ``screen_max_mean_abs_ppm`` and ``screen_min_valid_fraction``
   from the batch-level stability table.

The pipeline config supports the same flag:

.. code-block:: python

   from mioXpektron import PipelineConfig, run_pipeline

   config = PipelineConfig(auto_tune=True)
   intensity_df, area_df = run_pipeline(files, config=config)

This estimates ``mz_tolerance`` from median channel spacing and
``normalization_target`` from the median raw TIC.

The flat-window scanner and denoise evaluator also accept the flag:

.. code-block:: python

   from mioXpektron import ScanForFlatRegion
   scanner = ScanForFlatRegion(files=my_files, auto_tune=True)
   scanner.run()

.. code-block:: python

   from mioXpektron.denoise import compare_methods_in_windows
   rollup, summary, detail = compare_methods_in_windows(
       x, y, windows,
       auto_tune=True,
       auto_tune_files=my_files,
   )

Using Individual Estimators
---------------------------

Each estimator can be called directly if you want fine-grained control:

.. code-block:: python

   from mioXpektron.adaptive import (
       estimate_autodetect_tolerance,
       estimate_outlier_threshold,
       estimate_screening_thresholds,
       estimate_multisegment_breakpoints,
       estimate_normalization_target,
       estimate_mz_tolerance,
       estimate_flat_params,
       estimate_denoise_params,
       estimate_bootstrap_heuristics,
       auto_tune_calib_config,
   )

   # Calibration tolerance from peak widths
   tol_da = estimate_autodetect_tolerance(files, reference_masses)

   # Outlier threshold from residual distribution
   import numpy as np
   threshold = estimate_outlier_threshold(np.array(ppm_residuals))

   # Screening thresholds from stability summary
   screen = estimate_screening_thresholds(stability_df)

   # Multisegment breakpoints from mass distribution
   bps = estimate_multisegment_breakpoints(reference_masses, n_segments=3)

   # Pipeline parameters
   norm_target = estimate_normalization_target(files)
   mz_tol = estimate_mz_tolerance(files)

   # Flat-window parameters
   flat_overrides = estimate_flat_params(files)

   # Denoise parameters
   denoise_overrides = estimate_denoise_params(files)

   # Bootstrap heuristic overrides
   bootstrap_ov = estimate_bootstrap_heuristics(files)

   # Build a complete config with data-driven values
   config = auto_tune_calib_config(files, reference_masses)

Estimator Reference
-------------------

.. automodule:: mioXpektron.adaptive
   :members:
   :undoc-members:
   :show-inheritance:
