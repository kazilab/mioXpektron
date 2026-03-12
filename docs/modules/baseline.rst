Baseline Correction
===================

The baseline module removes broad background signals from ToF-SIMS spectra.
It wraps the `pybaselines <https://pybaselines.readthedocs.io/>`_ library and
adds batch processing, method evaluation, and flexible column name handling.

Quick Example
-------------

.. code-block:: python

   from mioXpektron import baseline_correction

   corrected = baseline_correction(intensity, method="airpls")

Available Methods
-----------------

mioXpektron supports all methods from pybaselines, organized by family:

**Polynomial methods**
   ``"poly"``, ``"modpoly"``, ``"imodpoly"``, ``"penalized_poly"``

**Whittaker methods**
   ``"asls"``, ``"iasls"``, ``"airpls"``, ``"arpls"``, ``"drpls"``,
   ``"iarpls"``, ``"aspls"``

**Morphological methods**
   ``"mor"``, ``"imor"``, ``"amormol"``, ``"rolling_ball"``

**Smoothing methods**
   ``"snip"``, ``"noise_median"``, ``"ipsa"``

**Miscellaneous**
   ``"dietrich"``, ``"golotvin"``, ``"std_distribution"``

Each method accepts its own keyword arguments, which are passed through to the
underlying pybaselines implementation.

Batch Baseline Correction
--------------------------

Process multiple files in parallel:

.. code-block:: python

   from mioXpektron import BaselineBatchCorrector

   corrector = BaselineBatchCorrector(
       method="airpls",
       method_kwargs={"lam": 1e6},
       max_workers=4,
   )

   results = corrector.process(file_list)

Method Evaluation
-----------------

Systematically compare baseline methods using quality metrics:

.. code-block:: python

   from mioXpektron import BaselineMethodEvaluator, small_param_grid_preset

   evaluator = BaselineMethodEvaluator()
   best = evaluator.evaluate(data)

The evaluator computes six metrics:

- **RFZN** --- Residual Flatness in Zero-signal regions (Noise)
- **NAR** --- Negative Area Ratio (how much correction goes below zero)
- **SNR** --- Signal-to-Noise Ratio improvement
- **BBI** --- Baseline-Below-Input indicator
- **BR** --- Baseline Roughness
- **NBC** --- Negative Bin Count

Flat Region Detection
---------------------

Identify flat (signal-free) regions for parameter tuning:

.. code-block:: python

   from mioXpektron import ScanForFlatRegion, FlatParams, AggregateParams

   scanner = ScanForFlatRegion()
   flat_regions = scanner.scan(data)

Column Name Flexibility
-----------------------

The baseline module automatically recognizes common column naming conventions:

- **Channel**: ``channel``, ``chan``, ``ch``, ``index``, ``idx``
- **m/z**: ``m/z``, ``mz``, ``mass``, ``mass_to_charge``
- **Intensity**: ``intensity``, ``counts``, ``signal``, ``int``

Matching is case-insensitive.

API Reference
-------------

.. autofunction:: mioXpektron.baseline.baseline_correction

.. autoclass:: mioXpektron.baseline.BaselineMethodEvaluator
   :members:

.. autoclass:: mioXpektron.baseline.BaselineBatchCorrector
   :members:

.. autoclass:: mioXpektron.baseline.ScanForFlatRegion
   :members:
