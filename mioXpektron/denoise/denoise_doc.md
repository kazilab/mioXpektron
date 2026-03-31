# Denoising API Reference

This document describes the current denoising-selection workflow in
`mioXpektron`. It covers:

- the high-level `DenoisingMethods` API for single-spectrum, windowed, and
  cohort-level selection
- the scientific selection logic used to rank candidate denoisers
- the `BatchDenoising` runner for writing denoised spectra to disk
- the lower-level `noise_filtering` routine for applying one chosen method

The current recommended entry point for method selection is
`DenoisingMethods.compare_across_files(...)`, not a single arbitrary spectrum.

---

## Selection Workflow

The denoising selector now uses a constrained, peak-centric workflow:

1. Detect reference peaks on the raw spectrum.
2. Re-measure those peaks after denoising using true local maxima only.
3. Compute peak-preservation metrics, global noise reduction, and matched-peak
   local SNR change.
4. Apply explicit pass/fail criteria before ranking methods.
5. Among passing methods, use the constrained Pareto rule to pick a default.

Important implementation details:

- Peak re-matching is conservative. A point inside the search window is not
  counted as a match unless it is a true detected local peak with acceptable
  prominence and width.
- The default candidate grid excludes derivative operators
  (`Savitzky-Golay deriv > 0`, Gaussian `order > 0`). Set
  `include_derivatives=True` only if you explicitly want them in the search.
- Windowed workflows aggregate across windows before ranking.
- Cohort workflows aggregate across files before ranking.

### Key ranking columns

The ranked summary returned by `rank_method(...)` and the high-level helpers
includes, among others:

- `frac_matched`
- `mz_shift_ppm`
- `mz_shift_iqr_ppm`
- `abs_height`
- `abs_area`
- `abs_fwhm`
- `noise_reduction_db`
- `delta_snr_db_med`
- `passes_peak_preservation`
- `passes_min_denoise`
- `passes_selection_criteria`
- `failed_criteria_count`
- `score`

### Default selection basis

`method_parameters(...)` and `select_methods(...)` default to
`basis="constrained_pareto_then_snr"`.

This means:

1. Keep only rows with `passes_selection_criteria == True` when
   `require_pass=True`.
2. Compute the Pareto frontier on:
   - minimize `abs_height`
   - maximize `delta_snr_db_med`
3. Pick the top method by:
   - highest `delta_snr_db_med`
   - then highest `frac_matched`
   - then lowest dimensionless `score`

If no methods pass and you still want the best exploratory candidate, set
`require_pass=False`.

### Default scientific gates

The current default gates are:

- `min_frac_matched = 0.85`
- `max_mz_shift_ppm = 10.0`
- `max_mz_shift_iqr_ppm = 10.0`
- `max_pct_area_med = 25.0`
- `max_pct_height_med = 25.0`
- `max_pct_fwhm_med = 25.0`
- `max_pct_area_iqr = 30.0`
- `max_pct_height_iqr = 30.0`
- `max_pct_fwhm_iqr = 30.0`

The denoising floor also requires:

- `noise_reduction_db >= 1.0`
- `delta_snr_db_med >= 1.0`

Override these through `selection_criteria={...}` in `compare(...)`,
`compare_in_windows(...)`, or `compare_across_files(...)`.

---

## DenoisingMethods

```python
class DenoisingMethods(mz_values, raw_intensities)
```

Evaluate, rank, and preview denoising strategies for mass-spectrometry data.

### Parameters

- `mz_values` : array-like of shape `(n_points,)`
  Spectrum axis (`m/z` or channel index).
- `raw_intensities` : array-like of shape `(n_points,)`
  Raw intensity values aligned with `mz_values`.

### Attributes

- `mz` : array-like of shape `(n_points,)`
  Stored spectrum axis.
- `intensity` : array-like of shape `(n_points,)`
  Stored raw intensities.

### Methods

```python
compare_across_files(
    file_paths,
    *,
    windows=None,
    min_mz=None,
    max_mz=None,
    per_window_max_peaks=50,
    min_prominence=None,
    search_ppm=20.0,
    match_min_prominence_ratio=0.1,
    match_min_prominence_abs=0.0,
    match_min_width_pts=0.25,
    resample_to_uniform=True,
    include_derivatives=False,
    return_format="pandas",
    selection_criteria=None,
    file_n_jobs=0,
    method_n_jobs=None,
    progress=True,
    save_summary=True,
)
```

Class method. Rank denoising methods across a cohort of spectra.

Use this when you want one default denoiser for a dataset rather than for one
single preview spectrum.

Returns:

- `ranked_summary`
  One ranked row per method across the cohort.
- `sample_summary_all`
  One row per sample and method.
- `detail_all`
  One row per measured peak and method, with `sample` and `source_file`
  columns added.

Notes:

- Use either `windows=...` or `min_mz` / `max_mz`, not both.
- File-level and method-level parallelism are both supported.
- With `file_n_jobs=0`, worker counts are chosen automatically to reduce
  oversubscription.

```python
compare(
    min_mz,
    max_mz,
    return_format="pandas",
    match_min_prominence_ratio=0.1,
    match_min_prominence_abs=0.0,
    match_min_width_pts=0.25,
    include_derivatives=False,
    selection_criteria=None,
    save_summary=True,
)
```

Rank denoising methods on one full spectrum range.

Returns:

- ranked summary table only

This is useful for exploratory work on one spectrum, but it is a weaker basis
for selecting a default denoiser than `compare_across_files(...)`.

```python
compare_in_windows(
    windows,
    per_window_max_peaks=50,
    min_prominence=None,
    search_ppm=20.0,
    match_min_prominence_ratio=0.1,
    match_min_prominence_abs=0.0,
    match_min_width_pts=0.25,
    resample_to_uniform=True,
    include_derivatives=False,
    return_format="pandas",
    selection_criteria=None,
    save_summary=True,
)
```

Rank denoising methods on one spectrum using multiple m/z windows.

Returns:

- ranked summary table aggregated across windows

The underlying implementation first aggregates per-window summaries, then ranks
methods on that rollup.

```python
plot(summary, annotate=True, top_k=3, save_plot=True, save_pareto=True)
```

Plot the Pareto comparison of `delta_snr_db_med` versus `abs_height`.

Returns:

- `matplotlib.axes.Axes`

Notes:

- By default this uses the same constrained basis as `method_parameters(...)`.
- If no methods pass the current criteria, the plot falls back to showing all
  finite candidates and labels the chart accordingly.

```python
denoise_check(
    denoise_params,
    *,
    sample_name="test",
    group=None,
    log_scale_y=False,
    mz_min=0,
    mz_max=500,
    show_peaks=False,
    peak_height=1000,
    peak_prominence=50,
    min_peak_width=1,
    max_peak_width=None,
    figsize=(10, 6),
    save_plot=True,
)
```

Apply one selected denoising configuration in memory and plot raw versus
denoised signal for the current `DenoisingMethods` instance.

Returns:

- `matplotlib.axes.Axes`

Note:

- This does not load a separate "corrected" file from disk. It uses
  `self.intensity` as the raw signal and computes the denoised trace on demand.

```python
method_parameters(
    summary,
    rank=0,
    basis="constrained_pareto_then_snr",
    require_pass=True,
    require_finite_metrics=True,
    save_selected=True,
)
```

Decode the parameter dictionary for a ranked method.

Returns:

- `dict`
  Arguments suitable for passing into `noise_filtering(...)`.

Notes:

- If `require_pass=True` and no rows pass the current gates, a clear error is
  raised explaining how to inspect exploratory candidates.
- Set `require_pass=False` if you want the best available candidate even when
  nothing satisfies all hard criteria.

### Example: cohort-level selection

```python
from pathlib import Path
from mioXpektron import DenoisingMethods
from mioXpektron.denoise.denoise_batch import load_txt_spectrum

files = sorted(Path("calibrated_spectra").glob("*_calibrated.txt"))

cohort_summary, per_file_summary, per_peak_detail = DenoisingMethods.compare_across_files(
    file_paths=files,
    min_mz=500.0,
    max_mz=520.0,
    include_derivatives=False,
    save_summary=True,
)

preview = load_txt_spectrum(files[0])
axis = preview.get("mz")
if axis is None or axis.size == 0:
    axis = preview.get("channel")

dm = DenoisingMethods(axis, preview["intensity"])
best_params = dm.method_parameters(
    cohort_summary,
    rank=0,
    basis="constrained_pareto_then_snr",
    require_pass=False,
)
dm.plot(cohort_summary, top_k=3)
dm.denoise_check(best_params, sample_name=files[0].stem, mz_min=500.0, mz_max=520.0)
```

### Example: stricter scientific gates

```python
summary, per_file_summary, per_peak_detail = DenoisingMethods.compare_across_files(
    file_paths=files,
    min_mz=500.0,
    max_mz=520.0,
    selection_criteria={
        "min_frac_matched": 0.90,
        "max_mz_shift_ppm": 5.0,
        "max_mz_shift_iqr_ppm": 5.0,
        "max_pct_height_med": 15.0,
        "max_pct_area_med": 15.0,
        "max_pct_fwhm_med": 15.0,
    },
)
```

---

## BatchDenoising

```python
class BatchDenoising(
    file_paths,
    *,
    method="wavelet",
    n_workers=None,
    backend="threads",
    progress=True,
    params=None,
)
```

Run `noise_filtering(...)` over many spectra and write denoised files to a
timestamped output folder.

### Parameters

- `file_paths` : iterable of path-like
  Input spectra to process.
- `method` : `{"wavelet", "gaussian", "median", "savitzky_golay", "none"}`
  Denoising method applied to every file.
- `n_workers` : int or None, default `None`
  Worker count for the batch run.
- `backend` : `{"threads", "processes"}`, default `"threads"`
  Execution backend.
- `progress` : bool, default `True`
  Toggle progress display.
- `params` : dict or None, default `None`
  Extra keyword arguments forwarded to `noise_filtering(...)`.

### Attributes

- `file_paths` : list of `Path`
- `method` : str
- `last_output_dir` : `Path | None`
- `last_results` : list of `BatchResult` or `None`

### Methods

```python
run(output_root, folder_name="denoised_spectrums", save_result=True)
```

Execute the batch denoising workflow.

Returns:

- list of `BatchResult`

### Example

```python
from mioXpektron.denoise import BatchDenoising

runner = BatchDenoising(
    ["data/spectrum_A.txt", "data/spectrum_B.txt"],
    method="wavelet",
    params={"variance_stabilize": "anscombe"},
)
results = runner.run("outputs/denoised")
print("Outputs stored in", runner.last_output_dir)
```

---

## noise_filtering

```python
noise_filtering(
    intensities,
    *,
    method="wavelet",
    window_length=15,
    polyorder=3,
    deriv=0,
    gauss_sigma_pts=None,
    gaussian_order=0,
    wavelet="sym8",
    level=None,
    threshold_strategy="universal",
    threshold_mode="soft",
    sigma=None,
    sigma_strategy="per_level",
    variance_stabilize="none",
    anscombe_negative_policy="warn_clip",
    cycle_spins=0,
    pywt_mode="periodization",
    clip_nonnegative=True,
    preserve_tic=False,
    x=None,
    resample_to_uniform=False,
    target_dx=None,
    forward_interp="pchip",
)
```

Apply one denoising or smoothing method to a 1D spectrum.

### Parameters

- `intensities` : ndarray of shape `(n_points,)`
  Input intensities.
- `method` : `{"savitzky_golay", "gaussian", "median", "wavelet", "none"}`
- `window_length` : int, default `15`
  Window size for Savitzky-Golay and median filters.
- `polyorder` : int, default `3`
  Polynomial order for Savitzky-Golay.
- `deriv` : int, default `0`
  Derivative order for Savitzky-Golay.
- `gauss_sigma_pts` : float or None, default `None`
  Gaussian sigma in points. If omitted, `window_length / 6` is used.
- `gaussian_order` : int, default `0`
  Derivative order for Gaussian filtering.
- `wavelet` : `{"db4", "db8", "sym5", "sym8", "coif2", "coif3"}`
- `level` : int or None, default `None`
  Wavelet decomposition depth.
- `threshold_strategy` : `{"universal", "bayes", "sure", "sure_opt"}`
- `threshold_mode` : `{"soft", "hard"}`
- `sigma` : float or None, default `None`
  Optional external noise estimate.
- `sigma_strategy` : `{"per_level", "global"}`
- `variance_stabilize` : `{"none", "anscombe"}`
- `anscombe_negative_policy` : `{"warn_clip", "clip", "raise"}`
- `cycle_spins` : `{0, 4, 8, 16, 32}`, default `0`
- `pywt_mode` : str, default `"periodization"`
- `clip_nonnegative` : bool, default `True`
- `preserve_tic` : bool, default `False`
- `x` : ndarray or None, default `None`
  Optional axis for non-uniform sampling.
- `resample_to_uniform` : bool, default `False`
  Resample to a uniform grid when `x` is provided.
- `target_dx` : float or None, default `None`
  Grid spacing for resampling.
- `forward_interp` : `{"pchip", "linear"}`

### Returns

- `denoised` : ndarray of shape `(n_points,)`

### Notes

- Non-finite samples are ignored during filtering and restored afterward.
- `method="none"` returns the input array after optional clipping or TIC
  preservation.
- Derivative filters are still available here, but they are not part of the
  default model-selection grid unless `include_derivatives=True`.

### Example

```python
from mioXpektron.denoise import noise_filtering

denoised = noise_filtering(
    intensities,
    method="wavelet",
    threshold_strategy="bayes",
    cycle_spins=8,
    variance_stabilize="anscombe",
    anscombe_negative_policy="raise",
)
```
