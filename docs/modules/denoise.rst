Denoising
=========

The denoise module provides multiple noise reduction strategies for ToF-SIMS
spectra, with tools for method comparison and batch processing.

Quick Example
-------------

.. code-block:: python

   from mioXpektron import noise_filtering

   denoised = noise_filtering(intensity, method="wavelet")

Available Methods
-----------------

Wavelet Denoising
^^^^^^^^^^^^^^^^^

The default and most powerful method. Uses discrete wavelet transforms with
configurable threshold strategies:

.. code-block:: python

   denoised = noise_filtering(
       intensity,
       method="wavelet",
       wavelet="db4",
       level=None,           # auto-select decomposition level
       threshold_strategy="sure",  # sure | bayes | visushrink | universal
   )

Features:

- Automatic decomposition level selection
- Multiple threshold strategies (SURE, BayesShrink, VisuShrink, Universal)
- Variance-Stabilizing Transform (VST) for Poisson-like noise
- Cycle-spinning for shift-invariant denoising

Gaussian Filter
^^^^^^^^^^^^^^^

.. code-block:: python

   denoised = noise_filtering(intensity, method="gaussian", sigma=1.0)

Median Filter
^^^^^^^^^^^^^

.. code-block:: python

   denoised = noise_filtering(intensity, method="median", kernel_size=5)

Savitzky-Golay Filter
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   denoised = noise_filtering(
       intensity,
       method="savitzky_golay",
       window_length=11,
       poly_order=3,
   )

Method Comparison
-----------------

Compare methods side by side to choose the best approach for your data:

.. code-block:: python

   from mioXpektron import compare_denoising_methods

   results = compare_denoising_methods(
       data,
       methods=["wavelet", "gaussian", "savgol"],
       metric="snr",
   )

Rank methods using Pareto-front analysis:

.. code-block:: python

   from mioXpektron import rank_method, select_methods

   rankings = rank_method(comparison_results)
   best = select_methods(rankings)

Batch Denoising
---------------

Process multiple files with parallel execution:

.. code-block:: python

   from mioXpektron import BatchDenoising, batch_denoise

   # Class-based interface
   denoiser = BatchDenoising(method="wavelet")
   results = denoiser.process_directory("data/")

   # Functional interface
   results = batch_denoise(file_list, method="wavelet", max_workers=4)

API Reference
-------------

.. autofunction:: mioXpektron.denoise.noise_filtering

.. autoclass:: mioXpektron.denoise.DenoisingMethods
   :members:

.. autoclass:: mioXpektron.denoise.BatchDenoising
   :members:

.. autofunction:: mioXpektron.denoise.compare_denoising_methods

.. autofunction:: mioXpektron.denoise.rank_method

.. autofunction:: mioXpektron.denoise.select_methods
