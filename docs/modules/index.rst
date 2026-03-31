Module Overview
===============

mioXpektron is organized into focused modules that can be used independently
or combined through the :doc:`../pipeline`.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Description
   * - :doc:`adaptive`
     - Data-driven estimation of pipeline parameters (opt-in ``auto_tune``)
   * - :doc:`baseline`
     - Baseline correction using 20+ algorithms, batch correction, and method evaluation
   * - :doc:`denoise`
     - Wavelet, Gaussian, median, and Savitzky-Golay denoising with method comparison
   * - :doc:`detection`
     - Peak detection (local max and CWT), area integration, cross-sample alignment
   * - :doc:`calibration`
     - Channel-to-m/z calibration with linear, quadratic, and reflectron TOF models
   * - :doc:`normalization`
     - 18 normalization methods including robust SNV, multi-ion reference, mass-stratified PQN, and method evaluation
   * - :doc:`plotting`
     - Publication-ready spectrum and peak visualization
   * - :doc:`utils`
     - File I/O, data import, batch processing, and statistical analysis

Data Flow
---------

A typical mioXpektron workflow follows this data flow::

   Raw Spectrum Files
        |
        v
   [Optional] Adaptive Parameter Estimation (auto_tune=True)
        |
        v
   [Optional] Auto-Calibration (Channel -> m/z)
        |
        v
   Denoising (wavelet / gaussian / median / savitzky_golay)
        |
        v
   Baseline Correction (AirPLS / AsLS / ...)
        |
        v
   Normalization (TIC / Poisson / SNV / PQN / ...)
        |
        v
   Peak Detection (local max or CWT)
        |
        v
   Cross-Sample m/z Alignment
        |
        v
   Output: Intensity & Area Matrices

.. toctree::
   :hidden:

   adaptive
   baseline
   denoise
   detection
   calibration
   normalization
   plotting
   utils
