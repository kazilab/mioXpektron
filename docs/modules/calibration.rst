Calibration
===========

The recalibrate module converts raw channel-based ToF-SIMS spectra to
calibrated m/z axes. It supports multiple Time-of-Flight models and provides
both automatic and manual calibration workflows.

Quick Example
-------------

.. code-block:: python

   from mioXpektron import AutoCalibrator, AutoCalibConfig

   config = AutoCalibConfig(
       reference_masses=[12.0, 28.0, 56.0],
       model="quadratic",
   )

   calibrator = AutoCalibrator(config)
   calibrated = calibrator.calibrate(spectra_files)

TOF Models
----------

mioXpektron supports three Time-of-Flight calibration models:

**Linear**
   ``m/z = a * channel + b``

   Simplest model. Suitable when the mass range is narrow or the instrument
   has been pre-calibrated.

**Quadratic**
   ``m/z = a * channel^2 + b * channel + c``

   Handles moderate non-linearity. Good default choice for most instruments.

**Reflectron**
   A physics-based model that accounts for the ion mirror geometry in
   reflectron-type TOF analyzers.

AutoCalibrator
--------------

Fully automatic calibration using known reference masses:

.. code-block:: python

   from mioXpektron import AutoCalibrator, AutoCalibConfig

   config = AutoCalibConfig(
       reference_masses=[1.008, 22.99, 38.96, 58.07, 86.10, 184.07],
       model="quadratic",
       output_folder="calibrated/",
       max_workers=4,
   )

   calibrator = AutoCalibrator(config)
   results = calibrator.calibrate(file_list, channel_dict)

The calibrator:

1. Picks peaks near expected channel positions
2. Matches them to reference masses
3. Fits the selected TOF model
4. Applies the calibration to the full spectrum
5. Writes calibrated spectra to the output folder

FlexibleCalibrator
------------------

For more control over the calibration process:

.. code-block:: python

   from mioXpektron import FlexibleCalibrator, FlexibleCalibConfig

   config = FlexibleCalibConfig(
       model="quadratic",
       outlier_threshold=3.0,
   )

   calibrator = FlexibleCalibrator(config)
   mz_calibrated = calibrator.calibrate(channels, known_channels, known_masses)

Features:

- Multiple peak-picking methods (max, centroid, parabolic, Gaussian, Voigt)
- Outlier detection using Huber regression
- PPM and Dalton error reporting
- Iterative refinement with residual analysis

Debugging Calibration
---------------------

Use the debug calibrator for detailed diagnostic output:

.. code-block:: python

   from mioXpektron import FlexibleCalibratorDebug, FlexibleCalibConfigDebug

   config = FlexibleCalibConfigDebug(model="quadratic")
   calibrator = FlexibleCalibratorDebug(config)

   # Produces detailed logs of each calibration step
   result = calibrator.calibrate(channels, known_channels, known_masses)

The debug version logs:

- Peak picking decisions at each reference mass
- Model fit residuals and quality metrics
- Outlier detection and removal steps
- Final calibration coefficients and errors

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
