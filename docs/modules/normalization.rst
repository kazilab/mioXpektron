Normalization
=============

The normalization module provides 14 normalization strategies for ToF-SIMS
spectra, ranging from simple scaling (TIC, max) to variance-stabilising
transforms (Poisson, sqrt, VSN) and robust methods (median, PQN).  An
evaluation framework helps choose the best method for your dataset.

Quick Example
-------------

.. code-block:: python

   from mioXpektron import normalize

   # TIC normalization (default)
   normalized = normalize(intensity, method="tic", target_tic=1e6)

   # Or use any of 14 available methods
   normalized = normalize(intensity, method="poisson")

Available Methods
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Method
     - Name
     - Use Case
   * - ``tic``
     - Total Ion Current
     - General-purpose normalization (default)
   * - ``median``
     - Median scaling
     - Robust when dominant peaks inflate TIC (e.g. substrate ions)
   * - ``rms``
     - Root Mean Square
     - Compromise between TIC and median sensitivity
   * - ``max``
     - Maximum normalization
     - Quick comparison (max value = 1)
   * - ``vector``
     - L2 vector norm
     - Spectral shape comparison (unit length)
   * - ``snv``
     - Standard Normal Variate
     - Before PCA/PLS-DA; removes multiplicative scatter
   * - ``poisson``
     - Poisson scaling
     - Before PCA on count data; equalises channel weights
   * - ``sqrt``
     - Square-root transform
     - Variance stabilisation for Poisson-distributed counts
   * - ``log``
     - Log(1+x) transform
     - High dynamic range spectra
   * - ``vsn``
     - arcsinh transform
     - Variance stabilisation; handles zeros gracefully
   * - ``minmax``
     - Min-Max scaling
     - Fixed range [0, 1]
   * - ``selected_ion``
     - Single-peak reference
     - Normalize to a known reference ion
   * - ``pqn``
     - Probabilistic Quotient
     - Compositional effects; requires dataset reference
   * - ``median_of_ratios``
     - DESeq2-style
     - Multi-batch experiments; requires geometric-mean reference

Method Details
^^^^^^^^^^^^^^

**TIC Normalization** --- Scales each spectrum so the sum of all intensities
equals a target value (default 1 million):

.. code-block:: python

   from mioXpektron import tic_normalization

   normalized = tic_normalization(intensity, target_tic=1e6)

**Poisson Scaling** --- Divides by ``sqrt(mean_intensity)``, equalising
the weight of low- and high-count channels.  Nearly universal for
multivariate analysis of ToF-SIMS count data:

.. code-block:: python

   from mioXpektron import normalize

   scaled = normalize(intensity, method="poisson")

**Selected-Ion Normalization** --- Normalizes to a reference peak (by index
or absolute intensity):

.. code-block:: python

   # By index
   normalized = normalize(intensity, method="selected_ion", reference_idx=42)

   # By absolute value
   normalized = normalize(intensity, method="selected_ion",
                          reference_intensity=5000.0)

**PQN (Probabilistic Quotient Normalization)** --- Handles compositional
effects. Requires a reference spectrum (e.g. median of the dataset):

.. code-block:: python

   import numpy as np
   from mioXpektron import normalize

   # Compute reference from all spectra
   reference = np.median(all_spectra, axis=0)
   normalized = normalize(intensity, method="pqn", reference=reference)


Unified Dispatcher
------------------

All methods are accessible through the :func:`normalize` function:

.. code-block:: python

   from mioXpektron import normalize, normalization_method_names

   # List available methods
   print(normalization_method_names())
   # ['log', 'max', 'median', 'median_of_ratios', 'minmax', 'poisson',
   #  'pqn', 'rms', 'selected_ion', 'snv', 'sqrt', 'tic', 'vector', 'vsn']

   # Apply any method
   result = normalize(intensity, method="rms", target_rms=1.0)


Method Evaluation
-----------------

The :class:`NormalizationEvaluator` compares methods on your actual data using
unsupervised, supervised, and spectral-quality metrics --- following the
approach from `xpectrass` adapted for ToF-SIMS:

.. code-block:: python

   from mioXpektron import NormalizationEvaluator

   evaluator = NormalizationEvaluator(
       files=["spectra/*.txt"],
       methods=["tic", "median", "rms", "poisson", "sqrt", "vsn"],
   )

   # Run evaluation
   results = evaluator.evaluate()

   # Print ranked summary
   evaluator.print_summary()

   # Generate plots
   evaluator.plot()

**Metrics computed:**

- **CV of TIC** --- Coefficient of variation of total-ion current (lower = more
  consistent normalization)
- **Within-group SAM** --- Spectral Angle Mapper between technical replicates
  (lower = more similar shapes)
- **Within-group correlation** --- Mean Pearson correlation within groups
  (higher = better consistency)
- **Clustering** --- Adjusted Rand Index, NMI, silhouette, and stability via
  KMeans
- **Supervised** --- Macro F1 and balanced accuracy via stratified
  cross-validation (requires scikit-learn and ≥2 groups)

**Composite scores** (z-scored, higher = better):

- ``score_combined`` --- Balanced across all metric categories (recommended)
- ``score_unsupervised`` --- For unlabelled data
- ``score_supervised`` --- Classification-focused
- ``score_efficient`` --- Includes computation time


Visual Comparison
-----------------

Preview how different methods affect a single spectrum:

.. code-block:: python

   from mioXpektron import NormalizationMethods

   nm = NormalizationMethods(mz_values, raw_intensity)

   # Side-by-side comparison
   nm.compare_visual(
       methods=["tic", "median", "poisson", "sqrt"],
       mz_min=0, mz_max=200,
   )

   # Apply one method and visualise with peak overlay
   nm.normalize_and_check(
       method="poisson",
       show_peaks=True,
       mz_min=0, mz_max=200,
   )


Data Preprocessing
------------------

Combined import, filtering, and normalization in one step:

.. code-block:: python

   from mioXpektron import data_preprocessing

   sample_name, group, mz, normalized = data_preprocessing(
       "spectrum.txt",
       mz_min=1.0,
       mz_max=300.0,
       normalization_target=1e6,
   )


Batch Normalization
-------------------

Functional Interface
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mioXpektron import batch_tic_norm

   written_files = batch_tic_norm(
       "spectra/*.txt",
       output_dir="normalized/",
       normalization_target=1e6,
   )

Class-Based Interface
^^^^^^^^^^^^^^^^^^^^^

For more control with Polars-based parallel processing:

.. code-block:: python

   from mioXpektron import BatchTicNorm

   normalizer = BatchTicNorm(
       input_pattern="spectra/*.csv",
       output_dir="normalized/",
       normalization_target=1e6,
       n_workers=4,
   )

   # View TIC statistics before processing
   stats = normalizer.get_tic_statistics()

   # Run batch normalization
   output_files = normalizer.process()


Batch Evaluation
^^^^^^^^^^^^^^^^

Evaluate methods across an entire dataset:

.. code-block:: python

   from mioXpektron import NormalizationMethods

   evaluator = NormalizationMethods.evaluate(
       files=["spectra/*.txt"],
       methods=["tic", "median", "rms", "poisson", "sqrt", "vsn"],
       n_jobs=-1,  # use all CPUs
   )

   # The returned evaluator has .plot() and .print_summary()
   evaluator.plot()


API Reference
-------------

.. autofunction:: mioXpektron.normalization.normalize

.. autofunction:: mioXpektron.normalization.normalization_method_names

.. autofunction:: mioXpektron.normalization.tic_normalization

.. autofunction:: mioXpektron.normalization.median_normalization

.. autofunction:: mioXpektron.normalization.rms_normalization

.. autofunction:: mioXpektron.normalization.max_normalization

.. autofunction:: mioXpektron.normalization.vector_normalization

.. autofunction:: mioXpektron.normalization.snv_normalization

.. autofunction:: mioXpektron.normalization.poisson_scaling

.. autofunction:: mioXpektron.normalization.sqrt_normalization

.. autofunction:: mioXpektron.normalization.log_normalization

.. autofunction:: mioXpektron.normalization.vsn_normalization

.. autofunction:: mioXpektron.normalization.minmax_normalization

.. autofunction:: mioXpektron.normalization.selected_ion_normalization

.. autofunction:: mioXpektron.normalization.pqn_normalization

.. autofunction:: mioXpektron.normalization.median_of_ratios_normalization

.. autoclass:: mioXpektron.normalization.NormalizationEvaluator
   :members:

.. autoclass:: mioXpektron.normalization.NormalizationMethods
   :members:

.. autofunction:: mioXpektron.normalization.data_preprocessing

.. autofunction:: mioXpektron.normalization.batch_tic_norm

.. autoclass:: mioXpektron.normalization.BatchTicNorm
   :members:
