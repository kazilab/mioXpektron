# mioXpektron

A comprehensive Time-of-Flight Secondary Ion Mass Spectrometry (ToF-SIMS) data processing toolkit for advanced signal processing, peak detection, and calibration.

## Features

mioXpektron provides a complete pipeline for ToF-SIMS data analysis:

### Core Modules

- **Baseline Correction** - Multiple baseline correction algorithms including AirPLS, AsLS, and adaptive methods
- **Denoising** - Advanced noise filtering strategies: wavelet transforms, Gaussian filters, median filters, and Savitzky-Golay smoothing
- **Peak Detection** - Robust peak detection with automatic noise estimation and overlapping peak resolution
- **Calibration** - Flexible mass spectrum recalibration with multiple TOF models (Linear, Quadratic, Reflectron)
- **Normalization** - TIC (Total Ion Current) normalization and data preprocessing
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

# Import your ToF-SIMS data
data = mx.import_data("path/to/your/data.csv")

# Denoise the spectrum
denoised = mx.noise_filtering(data, method='wavelet')

# Correct baseline
corrected = mx.baseline_correction(denoised, method='airpls')

# Detect peaks
peaks = mx.detect_peaks_with_area(corrected, snr_threshold=3.0)

# Visualize results
mx.PlotPeak(corrected, peaks)
```

### Automated Pipeline

```python
from mioXpektron import run_pipeline, PipelineConfig

# Configure the pipeline
config = PipelineConfig(
    denoise_method='wavelet',
    baseline_method='airpls',
    peak_detection_snr=3.0,
    normalize=True
)

# Run end-to-end processing
results = run_pipeline("path/to/data.csv", config)
```

### Calibration

```python
from mioXpektron import AutoCalibrator, AutoCalibConfig

# Automatic calibration with known reference peaks
config = AutoCalibConfig(
    reference_masses=[12.0, 28.0, 56.0],  # Known peaks
    model='quadratic'
)

calibrator = AutoCalibrator(config)
calibrated_data = calibrator.calibrate(data)
```

### Batch Processing

```python
from mioXpektron import BatchDenoising, batch_tic_norm

# Batch denoising
batch_denoiser = BatchDenoising(method='savgol', window_length=11)
denoised_files = batch_denoiser.process_directory("data/")

# Batch normalization
normalized = batch_tic_norm("data/", output_dir="normalized/")
```

## Advanced Features

### Method Comparison

Compare different denoising methods to find the optimal approach:

```python
from mioXpektron import compare_denoising_methods

results = compare_denoising_methods(
    data,
    methods=['wavelet', 'gaussian', 'savgol'],
    metric='snr'
)
```

### Baseline Evaluation

Systematically evaluate baseline correction methods:

```python
from mioXpektron import BaselineMethodEvaluator

evaluator = BaselineMethodEvaluator()
best_method = evaluator.evaluate(data)
```

### Overlapping Peak Resolution

Detect and analyze overlapping peaks:

```python
from mioXpektron import check_overlapping_peaks2

overlaps = check_overlapping_peaks2(peaks, resolution_threshold=0.5)
```

## Documentation

For detailed documentation on each module:

- **Denoising**: See [denoise_doc.md](mioXpektron/denoise/denoise_doc.md)
- **Baseline Correction**: See [COLUMN_NAMING.md](mioXpektron/baseline/COLUMN_NAMING.md)
- **Calibration**: See [DEBUG_README.md](mioXpektron/recalibrate/DEBUG_README.md)

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

- Python 3.8 or higher

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
