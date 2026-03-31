Baseline Correction
===================

The baseline module removes broad background signals from ToF-SIMS spectra.
It wraps the `pybaselines <https://pybaselines.readthedocs.io/>`_ library and
adds batch processing, method evaluation, and flexible column name handling.

Quick Example
-------------

.. code-block:: python

   from mioXpektron import baseline_correction

   corrected, baseline = baseline_correction(
       intensity,
       method="airpls",
       lam=1e6,
       return_baseline=True,
   )

Available Methods
-----------------

mioXpektron supports the 1-D baseline methods exposed by ``pybaselines`` plus
two lightweight filters:

- ``"median_filter"``
- ``"adaptive_window"``
- ``"poly"`` as a convenience alias
- the available ``pybaselines`` methods returned by ``baseline_method_names()``

For the current method list in your environment:

.. code-block:: python

   from mioXpektron import baseline_method_names

   print(baseline_method_names())

Each method accepts its own keyword arguments, which are passed through to the
underlying implementation. Parameterized evaluator labels such as
``"aspls(lam=1000000.0)"`` can also be passed back into the baseline utilities.

Batch Baseline Correction
--------------------------

Process multiple files in parallel:

.. code-block:: python

   from mioXpektron import BaselineBatchCorrector

   corrector = BaselineBatchCorrector(
       in_dir="denoised_spectra",
       pattern="*.txt",
       method="airpls",
       method_kwargs={"lam": 1e6},
       n_jobs=4,
   )

   out_dir = corrector.run(out_root="output_files")

Method Evaluation
-----------------

Systematically compare baseline methods using quality metrics:

.. code-block:: python

   import glob
   import random
   from mioXpektron import BaselineMethodEvaluator, ScanForFlatRegion

   files = sorted(glob.glob("output_files/denoised_spectrums_*/*.txt"))
   sample = sorted(random.Random(42).sample(files, min(10, len(files))))

   windows = ScanForFlatRegion(files=sample).run()

   param_grid = {
       "pspline_lsrpls": [{"lam": 1e6}],
       "pspline_drpls": [{"lam": 1e6}],
       "pspline_iarpls": [{"lam": 1e6}],
       "pspline_arpls": [{"lam": 1e6}],
       "pspline_airpls": [{"lam": 1e6}],
       "aspls": [{"lam": 1e6}],
       "imodpoly": [{"poly_order": 3}],
   }

   evaluator = BaselineMethodEvaluator(
       files=sample,
       methods=list(param_grid),
       param_grid=param_grid,
       flat_windows=windows,
   )
   summary = evaluator.evaluate(n_jobs=4)
   best = summary["overall_best_spec"]
   print(best["label"], best["method"], best["kwargs"])
   evaluator.preview_overlay(
       file=sample[0],
       methods=[spec["label"] for spec in summary["overall_order_specs"][:3]],
   )

If ``param_grid`` is provided and ``methods`` is omitted, the evaluator uses
the grid keys as the candidate set. For large cohorts, evaluating a
representative random subset of spectra first is usually much faster than
scoring every file.

The evaluator computes six metrics:

- **RFZN** --- Residual Flatness in Zero-signal regions (Noise)
- **NAR** --- Negative Area Ratio (how much correction goes below zero)
- **SNR** --- Signal-to-Noise Ratio improvement
- **BBI** --- Baseline-Below-Input indicator
- **BR** --- Baseline Roughness
- **NBC** --- Negative Bin Count

Flat Region Detection
---------------------

Identify flat (signal-free) regions for baseline evaluation:

.. code-block:: python

   from mioXpektron import ScanForFlatRegion

   scanner = ScanForFlatRegion(files=sample)
   flat_regions = scanner.run()

Column Name Flexibility
-----------------------

The baseline module automatically recognizes common column naming conventions:

- **Channel**: ``channel``, ``chan``, ``ch``, ``index``, ``idx``
- **m/z**: ``m/z``, ``mz``, ``mass``, ``moverz``, ``m_over_z``
- **Intensity**: ``intensity``, ``counts``, ``signal``, ``y``, ``ion_counts``

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
