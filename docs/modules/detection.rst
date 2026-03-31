Peak Detection
==============

The detection module identifies peaks in processed spectra, computes areas by
integration, and aligns peaks across multiple samples.

Quick Example
-------------

.. code-block:: python

   from mioXpektron import detect_peaks_with_area

   peaks_df = detect_peaks_with_area(
       mz_values=mz,
       intensities=corrected,
       sample_name="sample_01",
       group="control",
       min_snr=3.0,
       noise_model="mz_binned",
   )

Detection Algorithms
--------------------

Local Maximum Detection
^^^^^^^^^^^^^^^^^^^^^^^

The default method. Finds peaks as local maxima above a noise-based threshold:

.. code-block:: python

   from mioXpektron import detect_peaks_with_area, detect_peaks_with_area_v2

   # Standard version
   peaks = detect_peaks_with_area(
       mz_values=mz,
       intensities=corrected,
       sample_name="sample_01",
       group="control",
       min_snr=3.0,
   )

   # Enhanced version with additional peak properties
   peaks = detect_peaks_with_area_v2(
       mz_values=mz,
       intensities=corrected,
       sample_name="sample_01",
       group="control",
       min_snr=3.0,
       noise_model="mz_binned",
       noise_bins=20,
   )

CWT-Based Detection
^^^^^^^^^^^^^^^^^^^^

Uses the Continuous Wavelet Transform for multi-scale peak detection,
which is more robust to varying peak widths:

.. code-block:: python

   from mioXpektron import detect_peaks_cwt_with_area

   peaks = detect_peaks_cwt_with_area(
       mz_values=mz,
       intensities=corrected,
       sample_name="sample_01",
       group="control",
       min_snr=3.0,
   )

Noise Estimation
----------------

Robust noise estimation using the Median Absolute Deviation (MAD) approach,
which excludes peak regions for accurate background noise measurement:

.. code-block:: python

   from mioXpektron import robust_noise_estimation

   median_noise, std_noise = robust_noise_estimation(corrected)

The default global thresholding path uses a Gaussian-equivalent MAD estimate
on positive intensities after masking the measured width of detected peaks plus
an additional point margin. This is a robust heuristic for thresholding, not a
full physical Poisson detector model.

For spectra whose background varies across the mass range, the detection entry
points also support ``noise_model="mz_binned"``. This estimates local
background statistics in m/z bins and interpolates them back to a per-point
threshold profile:

.. code-block:: python

   peaks = detect_peaks_with_area_v2(
       mz_values=mz,
       intensities=corrected,
       sample_name="sample_01",
       group="control",
       noise_model="mz_binned",
       noise_bins=20,
       noise_min_points=25,
   )

Available noise models:

- ``"global"``: one threshold for the full spectrum
- ``"mz_binned"``: interpolated m/z-dependent thresholds

For spectra with strong mass-dependent background changes, ``"mz_binned"`` is
the preferred choice. The global model remains useful as a fast default, but
its SNR interpretation should be treated as heuristic.

Area Integration
----------------

Peak areas are computed from peak widths and corrected baselines. The current
integration path:

- handles empty or invalid background regions defensively
- integrates on the true floating peak boundaries
- reports the area definition and integration method in the output table

Batch Peak Collection
---------------------

``collect_peak_properties_batch()`` runs the full preprocessing and peak
collection workflow across many spectra and forwards the detector-specific
options consistently:

.. code-block:: python

   peaks_df = collect_peak_properties_batch(
       files=file_list,
       method="Gaussian",
       min_intensity=5,
       min_snr=3.0,
       noise_model="mz_binned",
       noise_bins=20,
   )

For analytic fit methods that enable overlapping-peak deconvolution, the
current implementation now uses a conservative two-stage acceptance rule:

- nearby peaks must overlap on an adaptive width-based spacing criterion
- the two-Gaussian fit must improve BIC over a single-Gaussian window fit by
  at least ``deconvolution_min_bic_delta`` (default ``10``)

Component widths are also checked against the configured peak-width bounds
before the deconvoluted peaks are accepted.

Cross-Sample Alignment
-----------------------

Align peaks across multiple samples by m/z tolerance:

.. code-block:: python

   from mioXpektron import align_peaks, PeakAlignIntensityArea

   # Align peak lists from multiple samples
   aligned = align_peaks(peak_list, mz_tolerance=0.2)

   # Full alignment with intensity and area matrices
   aligner = PeakAlignIntensityArea(
       mz_tolerance=0.2,
       method="Gaussian",
       noise_model="mz_binned",
       noise_bins=20,
       deconvolution_min_bic_delta=10.0,
   )
   intensity_matrix, area_matrix = aligner.align(peak_data)

``PeakAlignIntensityArea`` now exposes the underlying peak-detection method
and the same noise-model options as the batch collector, so alignment runs can
be compared on equal footing.

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

.. autofunction:: mioXpektron.detection.robust_noise_estimation_mz_dependent

.. autofunction:: mioXpektron.detection.collect_peak_properties_batch

.. autofunction:: mioXpektron.detection.align_peaks

.. autoclass:: mioXpektron.detection.PeakAlignIntensityArea
   :members:
