# Denoising API Reference

This document follows the scikit-learn documentation style for describing
estimators and utilities that operate on mass-spectrometry spectra. It
covers the high-level ``DenoisingMethods`` interface, the batch-oriented
``BatchDenoising`` runner, and the lower-level ``noise_filtering`` routine.

---

## DenoisingMethods

```python
class DenoisingMethods(mz, intensity)
```

Evaluate and visualize denoising strategies for a single spectrum.

### Parameters
- **mz** : array-like of shape (n_points,)
  Sample axis (m/z or channel numbers). Accepts NumPy arrays or Polars
  Series. Values must align with ``intensity``.
- **intensity** : array-like of shape (n_points,)
  Raw intensities corresponding to ``mz``.

### Attributes
- **mz_** : ndarray of shape (n_points,)
  Stored copy of the supplied axis.
- **intensity_** : ndarray of shape (n_points,)
  Stored copy of the supplied intensities.

### Methods

```python
compare(min_mz, max_mz, return_format='pandas', w_match=3.0, ...)
```
  Rank denoising methods on the full spectrum window ``[min_mz, max_mz]``.

  **Returns** : DataFrame-like
    Ranking table consistent with ``return_format``.

```python
compare_in_windows(windows, per_window_max_peaks=50, ...)
```
  Rank denoising methods across multiple user-specified ``(mz_min, mz_max)``
  windows.

  **Returns** : DataFrame-like
    Aggregated ranking for each method across all windows.

```python
plot(summary, annotate=True, top_k=3)
```
  Plot the Pareto front (delta SNR vs peak height) for the provided summary.

  **Returns** : matplotlib Axes
    Handle to the generated plot.

```python
denoise_check(denoise_params, sample_name='test', ...)
```
  Preview a single configuration by applying ``noise_filtering`` and plotting
  the raw vs denoised spectrum.

  **Returns** : matplotlib Axes
    Handle to the diagnostic plot.

```python
method_parameters(summary, rank=0)
```
  Decode the parameter dictionary for the method at the specified rank.

  **Returns** : dict
    Parsed ``noise_filtering`` configuration.

### Notes
- All comparison helpers internally call ``rank_method`` from
  ``denoise_select`` using the provided weighting factors.
- Intensities are not modified in place. Reuse the instance for multiple
  investigations on the same spectrum.

### Examples
```python
from mioXpektron.denoise import DenoisingMethods

dm = DenoisingMethods(mz, intensity)
summary = dm.compare(min_mz=100.0, max_mz=800.0)
top_params = dm.method_parameters(summary, rank=0)
dm.denoise_check(top_params, sample_name='Patient-01', log_scale_y=True)
```

---

## BatchDenoising

```python
class BatchDenoising(file_paths, *, method='wavelet', n_workers=None,
                     backend='threads', progress=True, params=None)
```

Run ``noise_filtering`` (through ``denoise_batch.batch_denoise``) on a list of
input spectra and write results to a timestamped directory.

### Parameters
- **file_paths** : iterable of path-like
  Files to process. Each entry is coerced to ``Path``. Must contain at least
  one path.
- **method** : {'wavelet', 'gaussian', 'median', 'savitzky_golay', 'none'}, default='wavelet'
  Denoising method applied to every file.
- **n_workers** : int or None, default=None
  Worker count. ``None`` or ``<= 0`` delegates selection to
  ``denoise_batch.batch_denoise``.
- **backend** : {'threads', 'processes'}, default='threads'
  Execution backend forwarded to the batch runner.
- **progress** : bool, default=True
  Toggle the progress bar.
- **params** : dict or None, default=None
  Extra keyword arguments forwarded to ``noise_filtering`` for each file.

### Attributes
- **file_paths** : list of ``Path``
  Normalised list of input files.
- **method** : str
  Stored method name.
- **last_output_dir** : Path or None
  Directory produced by the most recent ``run`` call.
- **last_results** : list of ``BatchResult`` or None
  Detailed results from the most recent ``run`` call.

### Methods

```python
run(output_root, folder_name='denoised_spectrums')
```
  Execute the batch denoising workflow.

  **Parameters**
  - **output_root** : path-like
    Destination directory. Created if missing.
  - **folder_name** : str, default='denoised_spectrums'
    Base folder name. A ``YYYYmmdd_HHMM`` timestamp is appended automatically.

  **Returns** : list of ``BatchResult``
    Mirrors the output of ``denoise_batch.batch_denoise``.

### Examples
```python
from mioXpektron.denoise import BatchDenoising

runner = BatchDenoising([
    "data/spectrum_A.txt",
    "data/spectrum_B.txt",
], method='wavelet', params={'variance_stabilize': 'anscombe'})
results = runner.run("outputs/denoised")
print("Outputs stored in", runner.last_output_dir)
```

---

## noise_filtering

```python
noise_filtering(intensities, *, method='wavelet', window_length=15,
                polyorder=3, deriv=0, gauss_sigma_pts=None,
                gaussian_order=0, wavelet='sym8', level=None,
                threshold_strategy='universal', threshold_mode='soft',
                sigma=None, sigma_strategy='per_level',
                variance_stabilize='none', cycle_spins=0,
                pywt_mode='periodization', clip_nonnegative=True,
                preserve_tic=False, x=None, resample_to_uniform=False,
                target_dx=None, forward_interp='pchip')
```

Apply a denoising or smoothing filter to a 1D spectrum.

### Parameters
- **intensities** : ndarray of shape (n_points,)
  Input intensities. Must be one-dimensional.
- **method** : {'savitzky_golay', 'gaussian', 'median', 'wavelet', 'none'}, default='wavelet'
- **window_length** : int, default=15
  Window size for Savitzky-Golay and median filters. Coerced to an odd integer.
- **polyorder** : int, default=3
  Polynomial order for Savitzky-Golay (``polyorder < window_length``).
- **deriv** : int, default=0
  Derivative order for Savitzky-Golay smoothing.
- **gauss_sigma_pts** : float or None, default=None
  Overrides the Gaussian kernel width (points). Defaults to ``window_length / 6``.
- **gaussian_order** : int, default=0
  Derivative order for Gaussian filtering.
- **wavelet** : {'db4', 'db8', 'sym5', 'sym8', 'coif2', 'coif3'}, default='sym8'
- **level** : int or None, default=None
  Decomposition depth for wavelet denoising. ``None`` selects a heuristic level.
- **threshold_strategy** : {'universal', 'bayes', 'sure', 'sure_opt'}, default='universal'
- **threshold_mode** : {'soft', 'hard'}, default='soft'
- **sigma** : float or None, default=None
  Optional noise estimate. ``None`` triggers automatic estimation.
- **sigma_strategy** : {'per_level', 'global'}, default='per_level'
- **variance_stabilize** : {'none', 'anscombe'}, default='none'
- **cycle_spins** : {0, 4, 8, 16, 32}, default=0
- **pywt_mode** : str, default='periodization'
- **clip_nonnegative** : bool, default=True
  Clip negative values in the final output.
- **preserve_tic** : bool, default=False
  Rescale the output to preserve the total ion current.
- **x** : ndarray of shape (n_points,) or None, default=None
  Optional axis for non-uniform samples.
- **resample_to_uniform** : bool, default=False
  Resample to a uniform grid when ``x`` is provided.
- **target_dx** : float or None, default=None
  Grid spacing used during resampling. ``None`` infers from ``x``.
- **forward_interp** : {'pchip', 'linear'}, default='pchip'
  Interpolation strategy used when resampling.

### Returns
- **denoised** : ndarray of shape (n_points,)
  Filtered intensities aligned with the input order.

### Raises
- **ValueError**
  If the inputs are not one-dimensional or shapes mismatch, or when filter
  parameters are inconsistent (for example ``polyorder < deriv``).

### Notes
- Non-finite samples are ignored and reinserted after filtering.
- When ``method='none'`` the function returns the input array after optional
  clipping/TIC conservation.
- The wavelet path delegates to ``wavelet_denoise`` for the actual shrinkage
  logic.

### Examples
```python
from mioXpektron.denoise import noise_filtering

denoised = noise_filtering(
    intensities,
    method='wavelet',
    threshold_strategy='bayes',
    cycle_spins=8,
    variance_stabilize='anscombe',
)
```

