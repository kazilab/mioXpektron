mioXpektron Documentation
=========================

**mioXpektron** is a comprehensive Time-of-Flight Secondary Ion Mass Spectrometry
(ToF-SIMS) data processing toolkit for advanced signal processing, peak detection,
and mass spectrum calibration.

It provides an end-to-end pipeline that takes raw ToF-SIMS spectra through
recalibration, denoising, baseline correction, normalization, peak detection,
and cross-sample alignment --- producing publication-ready intensity and area
matrices.

.. image:: https://img.shields.io/pypi/v/mioXpektron.svg
   :target: https://pypi.org/project/mioXpektron/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/mioXpektron.svg
   :alt: Python versions

.. image:: https://img.shields.io/github/license/kazilab/mioXpektron.svg
   :alt: License

Getting Started
---------------

Install mioXpektron and process your first spectrum in minutes:

.. code-block:: bash

   pip install mioXpektron

.. code-block:: python

   import mioXpektron as mx

   data = mx.import_data("spectrum.csv")
   denoised = mx.noise_filtering(data, method="wavelet")
   corrected = mx.baseline_correction(denoised, method="airpls")
   peaks = mx.detect_peaks_with_area(corrected, snr_threshold=3.0)

See the :doc:`quickstart` guide for a full walkthrough.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   pipeline
   modules/index

.. toctree::
   :maxdepth: 2
   :caption: Module Reference

   modules/baseline
   modules/denoise
   modules/detection
   modules/calibration
   modules/normalization
   modules/plotting
   modules/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   api

Acknowledgements
----------------

Parts of this documentation were created with assistance from ChatGPT Codex and Claude Code.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
