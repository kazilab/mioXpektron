Plotting
========

The plotting module provides publication-ready visualization tools for
ToF-SIMS spectra and peak analysis results.

Quick Example
-------------

.. code-block:: python

   from mioXpektron import PlotPeak

   PlotPeak(corrected_data, peaks_df)

Single Spectrum Plotting
------------------------

``PlotPeak`` creates annotated spectrum plots with detected peaks:

.. code-block:: python

   from mioXpektron import PlotPeak

   plot = PlotPeak(
       data,
       peaks,
       mz_min=1.0,
       mz_max=100.0,
       title="Sample Spectrum",
   )

Features:

- Automatic peak labeling with m/z values
- Configurable m/z range
- Peak prominence-based annotation filtering

Multi-Peak Visualization
------------------------

``PlotPeaks`` overlays spectra from multiple files for comparison:

.. code-block:: python

   from mioXpektron import PlotPeaks, PlotPeaksConfig

   config = PlotPeaksConfig(
       mz_min=20.0,
       mz_max=30.0,
       color_map={"cancer": "red", "control": "blue"},
   )

   PlotPeaks(file_list, config=config)

Group-based coloring is automatically inferred from filenames.

Overlapping Peak Plots
----------------------

Visualize and analyze overlapping peaks:

.. code-block:: python

   from mioXpektron import plot_overlapping_peaks

   plot_overlapping_peaks(peaks, data, resolution_threshold=0.5)

API Reference
-------------

.. autoclass:: mioXpektron.plotting.PlotPeak
   :members:

.. autoclass:: mioXpektron.plotting.PlotPeaks
   :members:

.. autoclass:: mioXpektron.plotting.PlotPeaksConfig
   :members:
   :undoc-members:
