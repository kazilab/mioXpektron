Peak Detection
==============

The detection module identifies peaks in processed spectra, computes areas by
integration, and aligns peaks across multiple samples.

Quick Example
-------------

.. code-block:: python

   from mioXpektron import detect_peaks_with_area

   peaks_df = detect_peaks_with_area(corrected_intensity, snr_threshold=3.0)

Detection Algorithms
--------------------

Local Maximum Detection
^^^^^^^^^^^^^^^^^^^^^^^

The default method. Finds peaks as local maxima above a noise-based threshold:

.. code-block:: python

   from mioXpektron import detect_peaks_with_area, detect_peaks_with_area_v2

   # Standard version
   peaks = detect_peaks_with_area(intensity, snr_threshold=3.0)

   # Enhanced version with additional peak properties
   peaks = detect_peaks_with_area_v2(intensity, snr_threshold=3.0)

CWT-Based Detection
^^^^^^^^^^^^^^^^^^^^

Uses the Continuous Wavelet Transform for multi-scale peak detection,
which is more robust to varying peak widths:

.. code-block:: python

   from mioXpektron import detect_peaks_cwt_with_area

   peaks = detect_peaks_cwt_with_area(intensity, min_snr=3.0)

Noise Estimation
----------------

Robust noise estimation using the Median Absolute Deviation (MAD) approach,
which excludes peak regions for accurate background noise measurement:

.. code-block:: python

   from mioXpektron import robust_noise_estimation

   noise_level = robust_noise_estimation(intensity)

Cross-Sample Alignment
-----------------------

Align peaks across multiple samples by m/z tolerance:

.. code-block:: python

   from mioXpektron import align_peaks, PeakAlignIntensityArea

   # Align peak lists from multiple samples
   aligned = align_peaks(peak_list, mz_tolerance=0.2)

   # Full alignment with intensity and area matrices
   aligner = PeakAlignIntensityArea(mz_tolerance=0.2)
   intensity_matrix, area_matrix = aligner.align(peak_data)

Overlapping Peak Analysis
-------------------------

Detect and visualize overlapping peaks:

.. code-block:: python

   from mioXpektron import check_overlapping_peaks, check_overlapping_peaks2

   # Basic overlap check
   overlaps = check_overlapping_peaks(peaks, resolution_threshold=0.5)

   # Enhanced analysis with visualization
   check_overlapping_peaks2(peaks, data, resolution_threshold=0.5)

API Reference
-------------

.. autofunction:: mioXpektron.detection.detect_peaks_with_area

.. autofunction:: mioXpektron.detection.detect_peaks_with_area_v2

.. autofunction:: mioXpektron.detection.detect_peaks_cwt_with_area

.. autofunction:: mioXpektron.detection.robust_peak_detection

.. autofunction:: mioXpektron.detection.robust_noise_estimation

.. autofunction:: mioXpektron.detection.align_peaks

.. autoclass:: mioXpektron.detection.PeakAlignIntensityArea
   :members:
