Changelog
=========

All notable changes to mioXpektron are documented here.

The format follows `Keep a Changelog <https://keepachangelog.com/>`_.

0.0.2 (2026-03-12)
------------------

**Changed**

- Synchronized package and documentation release metadata to ``0.0.2``
- Updated maintainer attribution to ``Data Analysis Team @KaziLab.se``
- Standardized contact metadata to ``mioxpektron@kazilab.se``

0.1.2 (2025)
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

0.1.1 (2025)
-------------

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

0.1.0 (2025)
-------------

**Added**

- Initial release
- Baseline correction with pybaselines integration
- Wavelet, Gaussian, median, and Savitzky-Golay denoising
- Local maximum and CWT peak detection with area integration
- TIC normalization
- Data import with auto-detection of file formats
- Basic spectrum plotting
