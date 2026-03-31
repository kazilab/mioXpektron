# mioXpektron

A comprehensive Time-of-Flight Secondary Ion Mass Spectrometry (ToF-SIMS) data processing toolkit for advanced signal processing, peak detection, and calibration.

## Features

mioXpektron provides a complete pipeline for ToF-SIMS data analysis:

### Core Modules

- **Baseline Correction** - Multiple baseline correction algorithms including AirPLS, AsLS, and adaptive methods
- **Denoising** - Advanced noise filtering strategies: wavelet transforms, Gaussian filters, median filters, and Savitzky-Golay smoothing
- **Peak Detection** - Robust peak detection with automatic noise estimation and overlapping peak resolution
- **Calibration** - Flexible mass spectrum recalibration with multiple TOF models, explicit autodetect fallback policies, bootstrap or m/z-based channel detection, and optional two-pass reference-mass screening
- **Normalization** - 18 normalization strategies including TIC, SNV, robust SNV, selected-ion, multi-ion reference, PQN, mass-stratified PQN, Pareto, and automated evaluation/ranking
- **Adaptive Parameterization** - Opt-in data-driven estimation of pipeline thresholds (`auto_tune=True`) for calibration tolerance, outlier rejection, normalization targets, and more
- **Visualization** - Publication-ready plotting tools for spectra and peak analysis
- **Batch Processing** - High-throughput data processing utilities
- **Pipeline** - End-to-end automated processing pipeline

## Installation

### From PyPI

```bash
pip install mioXpektron
```

### From Source

```bash
git clone https://github.com/kazilab/mioXpektron.git
cd mioXpektron
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
import mioXpektron as mx

# Import your ToF-SIMS data (.txt with m/z + Intensity or .csv with mz + corrected_intensity/intensity)
mz, intensity, sample_name, group = mx.import_data("path/to/your/data.txt")

# Denoise the spectrum
denoised = mx.noise_filtering(intensity, x=mz, method='wavelet')

# Correct baseline
corrected = mx.baseline_correction(denoised, method='airpls')

# Detect peaks
peaks = mx.detect_peaks_with_area(
    mz_values=mz,
    intensities=corrected,
    sample_name=sample_name,
    group=group,
    min_snr=3.0,
    noise_model="mz_binned",
)

# Visualize results
plot = mx.PlotPeak(
    mz_values=mz,
    raw_intensities=intensity,
    sample_name=sample_name,
    corrected_intensities=corrected,
)
plot.plot()
```

### Automated Pipeline

```python
import glob
from mioXpektron import run_pipeline, PipelineConfig

# Configure the pipeline
config = PipelineConfig(
    denoise_method='wavelet',
    baseline_method='airpls',
    normalization_target=1e6,
)

# Run end-to-end processing (returns intensity and area DataFrames)
files = glob.glob("path/to/files/*.txt")
intensity_df, area_df = run_pipeline(files, config=config)
```

### Adaptive Parameterization

Let the pipeline derive optimal parameters from your data instead of using
fixed defaults:

```python
from mioXpektron import FlexibleCalibrator, FlexibleCalibConfig

config = FlexibleCalibConfig(
    reference_masses=[1.0073, 27.0229, 29.0386, 41.0386, 57.0699, 104.1075],
    calibration_method="quad_sqrt",
    auto_tune=True,  # derives tolerance, screening, and breakpoints from data
)

calibrator = FlexibleCalibrator(config)
summary = calibrator.calibrate(files)
```

The pipeline also supports `auto_tune`:

```python
config = PipelineConfig(auto_tune=True)  # derives mz_tolerance and normalization_target
intensity_df, area_df = run_pipeline(files, config=config)
```

See `mioXpektron.adaptive` for individual estimator functions.

### Calibration

```python
import glob
from mioXpektron import FlexibleCalibrator, FlexibleCalibConfig

# Flexible calibration with screened reference masses
config = FlexibleCalibConfig(
    reference_masses=[1.0073, 27.0229, 29.0386, 41.0386, 57.0699, 104.1075],
    calibration_method="quad_sqrt",
    autodetect_method="parabolic",
    autodetect_fallback_policy="max",
    autodetect_strategy="mz",
    auto_screen_reference_masses=True,
    output_folder="calibrated_spectra",
)

calibrator = FlexibleCalibrator(config)
files = glob.glob("path/to/spectra/*.txt")
summary_df = calibrator.calibrate(files)
print(calibrator.last_reference_masses_used)
```

### Batch Processing

```python
import glob
from mioXpektron import BatchDenoising, batch_tic_norm

# Batch denoising
files = glob.glob("path/to/files/*.txt")
batch_denoiser = BatchDenoising(files, method='wavelet')
batch_denoiser.run(output_root="output_files", folder_name="denoised_spectrums")

# Batch TIC normalization (accepts a glob pattern)
output_paths = batch_tic_norm("data/*.txt", output_dir="normalized_spectra")
```

## Advanced Features

### Denoising Method Selection

The recommended workflow is cohort-level selection with
`DenoisingMethods.compare_across_files(...)`. This evaluates peak preservation
and denoising jointly, applies explicit pass/fail criteria, and by default
excludes derivative filters from the search grid.

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
    basis="constrained_pareto_then_snr",
    require_pass=False,
)

dm.plot(cohort_summary, top_k=3)
dm.denoise_check(best_params, sample_name=files[0].stem, mz_min=500.0, mz_max=520.0)
```

To tighten the scientific gates, pass `selection_criteria={...}` to
`compare(...)`, `compare_in_windows(...)`, or `compare_across_files(...)`.

### Baseline Evaluation

Systematically evaluate baseline correction methods across a set of spectra:

```python
from mioXpektron import BaselineMethodEvaluator

evaluator = BaselineMethodEvaluator(files=["data/*.txt"])
results_df = evaluator.evaluate()
evaluator.plot()
```

### Normalization Evaluation

Evaluate and rank normalization methods on labelled spectra:

```python
from mioXpektron import NormalizationEvaluator

evaluator = NormalizationEvaluator(
    files=["spectra/"],  # directory with .txt and/or baseline-corrected .csv spectra
    methods=["tic", "robust_snv", "pqn", "mass_stratified_pqn", "log"],
    method_kwargs_map={
        "mass_stratified_pqn": {
            "strata": [(0.0, 100.0), (100.0, 400.0), (400.0, float("inf"))],
        },
    },
)
results_df = evaluator.evaluate()
evaluator.plot()
evaluator.print_summary()
```

For cohort-level normalization on baseline-corrected CSV outputs, the repository
notebook `NoteBooks/_06_Normalization.ipynb` builds a shared m/z grid, supports
`linear`, `pchip`, `akima`, `makima` (SciPy >= 1.13), and `cubic`
interpolation, ranks normalization methods, previews overlays, and exports the
winning method. `mass_stratified_pqn` is included in the default notebook
evaluation; `multi_ion_reference` is available when users provide
`multi_ion_reference_mz` with optional `multi_ion_reference_values`.

### Overlapping Peak Resolution

Detect and visualize overlapping peaks across a dataset:

```python
from mioXpektron import check_overlapping_peaks2

check_overlapping_peaks2(
    data_dir="path/to/spectra",
    file_pattern="*.txt",
    mz_min=100.0,
    mz_max=200.0,
)
```

## Documentation

For detailed documentation on each module:

- **Adaptive Parameterization**: See [docs/modules/adaptive.rst](docs/modules/adaptive.rst)
- **Denoising**: See [denoise_doc.md](mioXpektron/denoise/denoise_doc.md)
- **Baseline Correction**: See [COLUMN_NAMING.md](mioXpektron/baseline/COLUMN_NAMING.md)
- **Calibration**: See [DEBUG_README.md](mioXpektron/recalibrate/DEBUG_README.md)
- **Sphinx docs**: See [docs/modules/calibration.rst](docs/modules/calibration.rst) and [docs/modules/detection.rst](docs/modules/detection.rst)

## Dependencies

- numpy >= 1.20.0
- pandas >= 1.3.0
- polars >= 0.18.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- PyWavelets >= 1.1.0
- pybaselines >= 1.0.0
- scikit-learn >= 1.0.0
- joblib >= 1.0.0
- tqdm >= 4.60.0

## Requirements

- Python 3.10 or higher

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Maintainer

- **Developed by**: Data Analysis Team @KaziLab.se
- **Contact**: mioxpektron@kazilab.se
- **Copyright**: @kazilab.se

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use mioXpektron in your research, please cite:

```bibtex
@software{mioxpektron,
  author = {Data Analysis Team @KaziLab.se},
  title = {mioXpektron: A ToF-SIMS Data Processing Toolkit},
  year = {2026},
  url = {https://github.com/kazilab/mioXpektron}
}
```

## Acknowledgments

mioXpektron builds upon established signal processing algorithms and the excellent scientific Python ecosystem.
Parts of this documentation were created with assistance from ChatGPT Codex and Claude Code.

## Support

For issues, questions, or contributions, please visit:
- **Issues**: https://github.com/kazilab/mioXpektron/issues
- **Documentation**: https://github.com/kazilab/mioXpektron#readme
