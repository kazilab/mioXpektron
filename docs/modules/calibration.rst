Calibration
===========

The recalibrate module converts raw channel-based ToF-SIMS spectra to
calibrated m/z axes. It supports multiple Time-of-Flight models and provides
both automatic and manual calibration workflows.

Current model families
----------------------

The recalibration backend supports the following model families:

- ``quad_sqrt``: empirical TOF model ``t = k*sqrt(m) + c*m + t0``
- ``linear_sqrt``: reduced two-parameter sqrt model
- ``poly2``: empirical quadratic calibration in channel space
- ``reflectron``: extended TOF model with a reflectron correction term
- ``spline``: non-parametric spline calibration
- ``multisegment``: piecewise ``quad_sqrt`` over user-defined mass ranges

``multisegment`` is available by explicit selection but remains experimental.
If you use it, choose breakpoints so every segment contains at least three
reference masses.

Quick Example
-------------

.. code-block:: python

   from mioXpektron import FlexibleCalibrator, FlexibleCalibConfig

   config = FlexibleCalibConfig(
       reference_masses=[1.0073, 27.0229, 29.0386, 41.0386, 57.0699, 104.1075],
       calibration_method="quad_sqrt",
       autodetect_method="parabolic",
       autodetect_fallback_policy="max",
       autodetect_strategy="mz",
       auto_screen_reference_masses=True,
   )

   calibrator = FlexibleCalibrator(config)
   summary = calibrator.calibrate(file_list)

AutoCalibrator
--------------

Fully automatic calibration using known reference masses:

.. code-block:: python

   from mioXpektron import AutoCalibrator, AutoCalibConfig

   config = AutoCalibConfig(
       reference_masses=[1.0073, 22.9892, 38.9632, 58.0657, 86.0970, 184.0733],
       model="quad_sqrt",
       autodetect_method="gaussian",
       autodetect_fallback_policy="max",
       autodetect_strategy="mz",
       output_folder="calibrated/",
       max_workers=4,
   )

   calibrator = AutoCalibrator(config)
   results = calibrator.calibrate(file_list)

The calibrator:

1. Auto-detects calibrant channels from either the ``m/z`` axis or a
   channel-only bootstrap path.
2. Applies the requested peak-picking method near each candidate calibrant.
3. Fits the selected model or, for ``AutoCalibrator``, compares the requested
   model set and keeps the best valid fit.
4. Applies the calibration to the full spectrum.
5. Writes calibrated spectra and summary tables to the output folder.

Autodetection modes
^^^^^^^^^^^^^^^^^^^

Both calibrators support two autodetection strategies:

- ``autodetect_strategy="mz"`` searches around the existing ``m/z`` axis.
- ``autodetect_strategy="bootstrap"`` reconstructs approximate channel
  positions directly from ``Channel`` and ``Intensity`` columns.

The bootstrap strategy now estimates both the slope and intercept of the
channel-to-mass relationship before searching locally for each calibrant, so
it is appropriate for spectra that do not yet have a trustworthy ``m/z`` axis.

Peak-picking methods
^^^^^^^^^^^^^^^^^^^^

The recalibration backend supports:

- ``max``
- ``centroid``
- ``centroid_raw``
- ``parabolic``
- ``gaussian``
- ``voigt``

Refined methods return fractional channel positions. ``centroid`` is now
baseline-aware and apex-focused; ``centroid_raw`` preserves the earlier
windowed center-of-mass behavior for comparison.

Fallback policy
^^^^^^^^^^^^^^^

``autodetect_fallback_policy`` controls what happens when a refined method
fails for a specific calibrant:

- ``"max"``: fall back to a robust local maximum pick
- ``"nan"``: keep the calibrant unresolved
- ``"raise"``: stop the run immediately

The actual method used per calibrant is recorded in
``calibrator.last_autodetect_methods``.

FlexibleCalibrator
------------------

For more control over the calibration process:

.. code-block:: python

   from mioXpektron import FlexibleCalibrator, FlexibleCalibConfig

   config = FlexibleCalibConfig(
       reference_masses=[1.0073, 27.0229, 29.0386, 41.0386, 57.0699, 104.1075],
       calibration_method="quad_sqrt",
       autodetect_method="parabolic",
       autodetect_fallback_policy="max",
       autodetect_strategy="mz",
       auto_screen_reference_masses=True,
       multisegment_breakpoints=[50, 150],
       outlier_threshold=3.0,
   )

   calibrator = FlexibleCalibrator(config)
   summary = calibrator.calibrate(file_list)

Features:

- Single-model calibration when you want to compare one model family directly
- Multiple peak-picking methods with explicit fallback control
- Outlier detection using Huber regression
- PPM and Dalton error reporting
- Optional two-pass reference-mass screening with per-mass residual summaries

Adaptive parameterization
^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``auto_tune=True`` to derive calibration parameters from the data:

.. code-block:: python

   config = FlexibleCalibConfig(
       reference_masses=reference_masses,
       calibration_method="quad_sqrt",
       auto_tune=True,
   )

   calibrator = FlexibleCalibrator(config)
   summary = calibrator.calibrate(file_list)

When ``auto_tune`` is active, the calibrator estimates:

- ``autodetect_tol_da`` from observed peak widths near calibrant masses
- ``multisegment_breakpoints`` from the calibrant mass distribution
- ``outlier_threshold`` from the residual distribution after the first fit pass
- ``screen_max_mean_abs_ppm`` and ``screen_min_valid_fraction`` from batch-level
  stability statistics

All estimates use sensible fallback values if insufficient data is available.
See :doc:`adaptive` for the individual estimator functions.

Reference-mass screening
^^^^^^^^^^^^^^^^^^^^^^^^

``FlexibleCalibrator`` can perform a fit-only first pass, summarize
calibrant stability, and refit using only stable reference masses:

.. code-block:: python

   config = FlexibleCalibConfig(
       reference_masses=reference_masses,
       calibration_method="quad_sqrt",
       auto_screen_reference_masses=True,
       screen_max_mean_abs_ppm=50.0,
       screen_min_valid_fraction=0.8,
       screen_min_count=3,
       screen_exclude_below_mz=1.5,
   )

   calibrator = FlexibleCalibrator(config)
   summary = calibrator.calibrate(file_list)

   print(calibrator.last_reference_masses_used)
   print(calibrator.last_reference_masses_screened_out)

This is useful when a small number of unstable tissue-specific anchors
dominate the overall calibration error.

Multisegment calibration
^^^^^^^^^^^^^^^^^^^^^^^^

``multisegment`` fits independent ``quad_sqrt`` models over mass intervals
defined by ``multisegment_breakpoints``. For example:

.. code-block:: python

   config = FlexibleCalibConfig(
       reference_masses=reference_masses,
       calibration_method="multisegment",
       multisegment_breakpoints=[50, 150],
   )

produces the segments ``0-50``, ``50-150``, and ``150-inf``.

Debugging Calibration
---------------------

Use the debug calibrator for detailed diagnostic output:

.. code-block:: python

   from mioXpektron import FlexibleCalibratorDebug, FlexibleCalibConfigDebug

   config = FlexibleCalibConfigDebug(calibration_method="quad_sqrt")
   calibrator = FlexibleCalibratorDebug(config)

   # Produces detailed logs of each calibration step
   result = calibrator.calibrate(channels, known_channels, known_masses)

The debug version logs:

- Peak picking decisions at each reference mass
- Model fit residuals and quality metrics
- Outlier detection and removal steps
- Final calibration coefficients and errors

Notebook workflow
-----------------

The calibration comparison notebook ``NoteBooks/_01_Calibration_Methods_Comparison.ipynb``
has been updated to expose:

- ``autodetect_fallback_policy``
- ``autodetect_strategy``
- ``centroid_raw``
- reference-mass screening settings
- per-spectrum autodetect diagnostics

When rerunning the notebook after backend changes, restart the kernel or rerun
the import cell so the local recalibration modules are reloaded explicitly.

API Reference
-------------

.. autoclass:: mioXpektron.recalibrate.AutoCalibrator
   :members:

.. autoclass:: mioXpektron.recalibrate.AutoCalibConfig
   :members:
   :undoc-members:

.. autoclass:: mioXpektron.recalibrate.FlexibleCalibrator
   :members:

.. autoclass:: mioXpektron.recalibrate.FlexibleCalibConfig
   :members:
   :undoc-members:
