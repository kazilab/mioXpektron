Changelog
=========

All notable changes to mioXpektron are documented here.

The format follows `Keep a Changelog <https://keepachangelog.com/>`_.

0.0.3 (2026-03-31)
-------------------

**Added**

- ``adaptive`` module with opt-in data-driven parameterization. New
  ``auto_tune=True`` flag on ``FlexibleCalibConfig``, ``PipelineConfig``,
  ``ScanForFlatRegion``, and ``compare_methods_in_windows()`` replaces
  hardcoded thresholds with values estimated from spectra.
- 10 individual estimator functions: ``estimate_autodetect_tolerance``,
  ``estimate_outlier_threshold``, ``estimate_screening_thresholds``,
  ``estimate_multisegment_breakpoints``, ``estimate_normalization_target``,
  ``estimate_mz_tolerance``, ``estimate_flat_params``,
  ``estimate_denoise_params``, ``estimate_bootstrap_heuristics``, and
  ``auto_tune_calib_config``.
- ``DEFAULT_REFERENCE_MASSES`` canonical 18-ion reference mass list exported
  from the pipeline module, replacing divergent inline fallback lists.
- Bootstrap heuristics in ``_models.py`` now accept an optional
  ``bootstrap_overrides`` dict to inject data-derived constants.
- ``autodetect_fallback_policy={"max", "nan", "raise"}`` for
  ``AutoCalibrator`` and ``FlexibleCalibrator`` so refined peak-picking
  failures can either fall back, return ``NaN``, or stop the run.
- ``centroid_raw`` as an explicit recalibration peak-picking mode for direct
  comparison against the new baseline-aware ``centroid`` implementation.
- Two-pass reference-mass screening in ``FlexibleCalibrator`` with explicit
  exclusions, per-mass stability summaries, and run diagnostics exposed via
  ``last_reference_masses_used``, ``last_reference_masses_screened_out``, and
  ``last_reference_mass_screening``.
- ``noise_model="mz_binned"`` for detection entry points and
  ``PeakAlignIntensityArea`` to support m/z-dependent noise thresholds.
- Per-spectrum autodetect diagnostics in the calibration comparison notebook,
  including actual methods used, fallback counts, and screening exports.

**Changed**

- Recalibration peak-picking methods now return fractional channel positions
  for refined fits instead of snapping all refined centers back to integer bins.
- ``centroid`` now uses baseline-aware local apex support; the previous raw
  local centroid is preserved as ``centroid_raw``.
- Bootstrap autodetection now estimates a channel mapping of the form
  ``channel = t0 + k*sqrt(m)`` and performs final peak selection in the raw
  local channel window, substantially improving agreement with the ``mz``
  autodetection path.
- ``quad_sqrt`` parameter validation is now local to the calibrated mass range,
  avoiding false rejections near ``H+``.
- ``PeakAlignIntensityArea`` now exposes the underlying detection method and
  noise-model options instead of hard-wiring a single detector configuration.
- Detection noise masking now excludes the measured peak width plus a
  configurable margin instead of using only a fixed point-count window.
- ``detect_peaks_with_area_v2()`` now uses the same shared noise-model helper
  as the other detection entry points, reducing method-to-method threshold
  inconsistencies.
- Overlapping-peak deconvolution in ``robust_peak_detection()`` now requires
  a BIC improvement over a single-peak fit and validates fitted component
  widths against the configured peak-width bounds before accepting a
  two-Gaussian solution.
- Analytic peak-fitting detection now emits one warning-level summary per
  spectrum when single-peak or deconvolution fits raise exceptions, instead of
  silently skipping all failed fit windows unless verbose debug logging is on.
- Wavelet ``variance_stabilize="anscombe"`` is now documented and enforced as
  the classical pure-Poisson Anscombe transform with the Mäkitalo-Foi unbiased
  inverse, and negative-input handling is now explicit via
  ``anscombe_negative_policy`` instead of silent clipping.
- Bootstrap autodetection heuristics now use named documented constants in the
  shared recalibration backend instead of unexplained inline literals.

**Fixed**

- Guarded robust noise estimation against empty or non-positive background
  samples so detection thresholds no longer silently become ``NaN``.
- Corrected Simpson baseline integration to use the true floating peak
  endpoints and updated the integration call for current SciPy expectations.
- Fixed ``collect_peak_properties_batch()`` to forward the user-provided
  ``min_intensity`` when using the default local-maximum detector.
- Fixed off-by-one slice limits in local peak windows and removed redundant
  ``locals()`` checks in combined detection paths.
- Fixed notebook result-table broadcasting for list-valued screening metadata
  and made the calibration comparison notebook reload local recalibration
  modules explicitly to avoid stale imports.

0.0.2 (2026-03-12)
------------------

**Changed**

- Synchronized package and documentation release metadata to ``0.0.2``
- Updated maintainer attribution to ``Data Analysis Team @KaziLab.se``
- Standardized contact metadata to ``mioxpektron@kazilab.se``

0.0.1 (2025)
-------------

**Added**

- 14 normalization methods: TIC, median, RMS, max, vector, SNV, Poisson,
  sqrt, log, VSN, MinMax, selected-ion, PQN, and median-of-ratios
- ``normalize()`` unified dispatcher for all normalization methods
- ``NormalizationEvaluator`` for data-driven method comparison using
  spectral-quality, clustering, and supervised metrics with composite scoring
- ``NormalizationMethods`` orchestrator for visual comparison and evaluation
- ``normalization_method_names()`` to list available methods

**Changed**

- Consolidated 3 inline TIC implementations to canonical ``tic_normalization()``
- Replaced ``multiprocessing.Pool`` with ``concurrent.futures.ProcessPoolExecutor``
  in ``BatchTicNorm`` for consistency with rest of codebase
- Standardised logging: converted ``print()`` to ``logging.getLogger(__name__)``
  across 13+ modules
- Refactored ``FlexibleCalibratorDebug`` to inherit from ``FlexibleCalibrator``
  (reduced ~1145 to ~280 lines)

0.0.0.post1 (2025)
------------------

**Added**

- End-to-end ``run_pipeline`` with ``PipelineConfig`` for batch processing
- ``FlexibleCalibrator`` and ``AutoCalibrator`` for mass spectrum calibration
- ``BaselineMethodEvaluator`` for systematic baseline method comparison
- ``BatchDenoising`` and ``batch_denoise`` for parallel denoising
- ``BatchTicNorm`` with Polars-based parallel normalization
- ``PeakAlignIntensityArea`` for cross-sample peak alignment
- ``PlotPeaks`` and ``PlotPeaksConfig`` for multi-sample visualization
- ``ScanForFlatRegion`` for automated flat region detection
- Column name alias system for flexible data input
- Debug calibrator with diagnostic logging
- Method comparison and Pareto-front ranking for denoising

0.0.0 (2025)
-------------

**Added**

- Initial release
- Baseline correction with pybaselines integration
- Wavelet, Gaussian, median, and Savitzky-Golay denoising
- Local maximum and CWT peak detection with area integration
- TIC normalization
- Data import with auto-detection of file formats
- Basic spectrum plotting
