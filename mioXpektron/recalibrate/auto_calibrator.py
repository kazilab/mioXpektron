"""
AutoCalibrator — automatic multi-model calibration for mass spectrometry.

Fits all requested calibration models, picks the best one per file, and
applies the winning model.  Model fitting, inversion, and peak-detection
live in the shared ``_models`` backend.

Author: Data Analysis Team @KaziLab.se
Version: 0.0.1
"""

import os
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- shared backend ---
from ._models import (
    MODEL_REGISTRY,
    _MODEL_ALIASES,
    _ModelMeta,
    _ppm_error,
    _ppm_to_da,
    _detect_outliers_huber,
    _estimate_noise_level,
    _fit_gaussian_peak,
    _fit_voigt_peak,
    _parabolic_peak_center,
    _enhanced_pick_channels,
    _enhanced_bootstrap_channels,
    _robust_initial_params_quad_sqrt,
    _fit_quad_sqrt_robust,
    _fit_reflectron,
    _fit_multisegment,
    _fit_spline_model,
    _fit_linear_sqrt,
    _fit_poly2,
    _invert_quad_sqrt,
    _invert_reflectron,
    _invert_linear_sqrt,
    _invert_poly2,
    _invert_spline,
    _invert_physical,
    apply_model_to_spectrum,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=RuntimeWarning)


class CalibrationModel(Enum):
    """Enumeration of available calibration models."""
    QUAD_SQRT = "quad_sqrt"
    LINEAR_SQRT = "linear_sqrt"
    POLY2 = "poly2"
    REFLECTRON = "reflectron"
    MULTISEGMENT = "multisegment"
    SPLINE = "spline"
    PHYSICAL = "physical"


class PeakDetectionMethod(Enum):
    """Enumeration of peak detection methods."""
    MAX = "max"
    CENTROID = "centroid"
    CENTROID_RAW = "centroid_raw"
    PARABOLIC = "parabolic"
    GAUSSIAN = "gaussian"
    VOIGT = "voigt"


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class AutoCalibConfig:
    """Universal calibration configuration with robust options.

    Parameters
    ----------
    reference_masses : list of float
        Known calibrant ion masses (m/z).
    model : str, optional
        Convenience shortcut — a single model name (or common alias like
        ``'quadratic'``, ``'tof'``, ``'linear'``).  Resolved into
        *models_to_try* during ``__post_init__``.  Ignored when
        *models_to_try* is explicitly provided.
    models_to_try : list of str, optional
        Explicit list of model names to fit.  Default: all production-ready
        models (excludes experimental ones such as ``multisegment`` and
        ``physical``).
    """

    reference_masses: List[float]
    output_folder: str = "calibrated_spectra"
    max_workers: Optional[int] = None

    # Peak detection parameters
    autodetect_tol_da: Optional[float] = None
    autodetect_tol_ppm: Optional[float] = None
    autodetect_method: str = "gaussian"
    autodetect_fallback_policy: str = "max"
    autodetect_strategy: str = "mz"
    prefer_recompute_from_channel: bool = False

    # Robust fitting parameters
    outlier_threshold: float = 3.0
    use_outlier_rejection: bool = True
    max_iterations: int = 3

    # Model selection parameters
    model: Optional[str] = None  # convenience alias -> models_to_try
    models_to_try: Optional[List[str]] = None
    prefer_physical_models: bool = True

    # Quality control
    min_calibrants: int = 3
    max_ppm_warning: float = 100.0
    max_ppm_error: float = 500.0

    # Advanced parameters
    use_bootstrap_init: bool = True
    spline_smoothing: Optional[float] = None
    multisegment_breakpoints: List[float] = field(default_factory=lambda: [50, 200, 500])

    # Instrument parameters (for physical model)
    instrument_params: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.autodetect_fallback_policy not in {"max", "nan", "raise"}:
            raise ValueError(
                "autodetect_fallback_policy must be 'max', 'nan', or 'raise'"
            )

        # Resolve the convenience ``model`` alias
        if self.models_to_try is None:
            if self.model is not None:
                canonical = _MODEL_ALIASES.get(self.model.lower(), self.model.lower())
                if canonical not in MODEL_REGISTRY:
                    raise ValueError(
                        f"Unknown model '{self.model}'.  "
                        f"Valid names: {sorted(MODEL_REGISTRY)}; "
                        f"aliases: {sorted(_MODEL_ALIASES)}"
                    )
                self.models_to_try = [canonical]
            else:
                # Default: all production-ready models (selection_allowed=True)
                self.models_to_try = [
                    name for name, meta in MODEL_REGISTRY.items()
                    if meta.selection_allowed
                ]

        # Warn if any requested model is experimental
        for name in self.models_to_try:
            meta = MODEL_REGISTRY.get(name)
            if meta and meta.experimental:
                logger.warning(
                    "Model '%s' is experimental: %s", name, meta.description
                )


# ---------------------------------------------------------------------------
#  Fit-and-score: tries all requested models on one file
# ---------------------------------------------------------------------------

def _fit_and_score_models_enhanced(
    filename: str,
    ref_masses: npt.NDArray[np.float64],
    calib_channels_dict: Dict[str, Sequence[float]],
    config: AutoCalibConfig,
) -> Optional[Tuple[str, Dict]]:
    """Fit all requested models and pick the best one for *filename*."""
    if filename not in calib_channels_dict:
        logger.warning(f"{filename}: Not found in calibration channels dict")
        return None

    t_meas = np.asarray(calib_channels_dict[filename], dtype=float)
    m_ref = np.asarray(ref_masses, dtype=float)

    valid = np.isfinite(t_meas)
    n_valid = valid.sum()

    if n_valid < config.min_calibrants:
        logger.warning(f"{filename}: Only {n_valid} valid calibrants (need >={config.min_calibrants})")
        return None

    t_meas = t_meas[valid]
    m_ref = m_ref[valid]

    logger.info(f"{filename}: Fitting with {n_valid} calibrants")

    out: Dict[str, Dict] = {}

    for model_name in config.models_to_try:
        try:
            if model_name == "quad_sqrt":
                params = _fit_quad_sqrt_robust(
                    m_ref, t_meas,
                    config.outlier_threshold,
                    config.max_iterations,
                )
                if params:
                    m_est = _invert_quad_sqrt(t_meas, *params)
                    ppm = _ppm_error(m_ref, m_est)
                    out[model_name] = {"params": params, "ppm": ppm, "n_calibrants": n_valid}

            elif model_name == "reflectron":
                params = _fit_reflectron(m_ref, t_meas)
                if params:
                    m_est = _invert_reflectron(t_meas, *params)
                    ppm = _ppm_error(m_ref, m_est)
                    out[model_name] = {"params": params, "ppm": ppm, "n_calibrants": n_valid}

            elif model_name == "multisegment":
                segment_data = _fit_multisegment(m_ref, t_meas, config.multisegment_breakpoints)
                if segment_data:
                    # Evaluate multisegment ppm by applying each segment to its calibrants
                    m_est_ms = np.full_like(m_ref, np.nan)
                    for seg_info in segment_data["segments"].values():
                        low, high = seg_info["range"]
                        seg_mask = (m_ref >= low) & (m_ref < high)
                        if seg_mask.any():
                            m_est_ms[seg_mask] = _invert_quad_sqrt(
                                t_meas[seg_mask], *seg_info["params"]
                            )
                    ppm = _ppm_error(m_ref, m_est_ms)
                    out[model_name] = {"params": segment_data, "ppm": ppm, "n_calibrants": n_valid}

            elif model_name == "spline":
                spline = _fit_spline_model(m_ref, t_meas, config.spline_smoothing)
                if spline:
                    m_est = _invert_spline(t_meas, spline)
                    ppm = _ppm_error(m_ref, m_est)
                    out[model_name] = {"params": spline, "ppm": ppm, "n_calibrants": n_valid}

            elif model_name == "linear_sqrt":
                params = _fit_linear_sqrt(m_ref, t_meas)
                if params:
                    m_est = _invert_linear_sqrt(t_meas, *params)
                    ppm = _ppm_error(m_ref, m_est)
                    out[model_name] = {"params": params, "ppm": ppm, "n_calibrants": n_valid}

            elif model_name == "poly2":
                params = _fit_poly2(m_ref, t_meas)
                if params:
                    m_est = _invert_poly2(t_meas, *params)
                    ppm = _ppm_error(m_ref, m_est)
                    out[model_name] = {"params": params, "ppm": ppm, "n_calibrants": n_valid}

            else:
                logger.debug(f"{filename}: No handler for model '{model_name}'; skipping")

        except Exception as e:
            logger.debug(f"{filename}: Failed to fit {model_name}: {e}")
            continue

    if not out:
        logger.warning(f"{filename}: All models failed to fit")
        return None

    # Select best model — only among models whose metadata allows auto-selection.
    selectable = {
        k: v for k, v in out.items()
        if MODEL_REGISTRY.get(k, _ModelMeta(k, True, False, True, "")).selection_allowed
    }
    if not selectable:
        selectable = out

    if config.prefer_physical_models:
        physical_models = ["quad_sqrt", "reflectron"]
        for model in physical_models:
            if model in selectable and selectable[model]["ppm"] < config.max_ppm_warning:
                best_name = model
                break
        else:
            best_name = min(selectable.keys(), key=lambda k: selectable[k]["ppm"])
    else:
        best_name = min(selectable.keys(), key=lambda k: selectable[k]["ppm"])

    out["best_model"] = best_name

    best_ppm = out[best_name]["ppm"]
    if best_ppm > config.max_ppm_error:
        logger.error(f"{filename}: Best model {best_name} exceeds max PPM ({best_ppm:.1f} > {config.max_ppm_error})")
        return None
    elif best_ppm > config.max_ppm_warning:
        logger.warning(f"{filename}: High PPM error ({best_ppm:.1f})")

    logger.info(f"{filename}: Best model={best_name}, ppm={best_ppm:.1f}")

    return filename, out


# ---------------------------------------------------------------------------
#  Apply calibration to a single file
# ---------------------------------------------------------------------------

def _apply_model_to_file_enhanced(args: Tuple[str, Dict[str, Dict], str]) -> Optional[str]:
    """Apply the best calibration model to a file."""
    file_path, best_map, out_folder = args
    fname = os.path.basename(file_path)

    if fname not in best_map:
        return None

    rec = best_map[fname]
    model = rec["best_model"]
    params = rec[model]["params"]

    try:
        df = pd.read_csv(file_path, sep="\t", header=0, comment="#")
    except Exception as e:
        logger.error(f"Failed to read {fname}: {e}")
        return None

    if "Channel" not in df.columns:
        logger.error(f"{fname}: Missing 'Channel' column")
        return None

    t = df["Channel"].astype(np.float64).to_numpy()

    mz_cal = apply_model_to_spectrum(t, model, params)

    out_df = pd.DataFrame({
        "m/z": mz_cal,
        "Intensity": df["Intensity"].to_numpy(),
        "Channel": t
    })

    os.makedirs(out_folder, exist_ok=True)
    out_fp = os.path.join(out_folder, fname.replace(".txt", "_calibrated.txt"))

    with open(out_fp, 'w') as f:
        f.write(f"# Calibration model: {model}\n")
        f.write(f"# Calibration error: {rec[model]['ppm']:.1f} ppm\n")
        f.write(f"# Number of calibrants: {rec[model]['n_calibrants']}\n")
        out_df.to_csv(f, sep="\t", index=False)

    return fname


# ---------------------------------------------------------------------------
#  Main calibrator class
# ---------------------------------------------------------------------------

class AutoCalibrator:
    """Automatic multi-model calibrator.

    Fits all requested models, selects the best one per file, and writes
    calibrated spectra.
    """

    def __init__(self, config: Optional[AutoCalibConfig] = None):
        default_masses = [
            1.00782503224,   # H+
            22.9897692820,   # Na+
            38.9637064864,   # K+
            58.065674,       # Organic fragment
            86.096974,       # Organic fragment
            104.107539,      # Organic fragment
            184.073871,      # Organic fragment
            224.105171       # Organic fragment
        ]

        self.config = config or AutoCalibConfig(reference_masses=default_masses)
        self.last_autodetect_methods: Dict[str, List[str]] = {}

        if len(self.config.reference_masses) < 3:
            raise ValueError("At least 3 reference masses required")

        logger.info(f"AutoCalibrator initialized with {len(self.config.models_to_try)} models")

    def calibrate(
        self,
        files: Sequence[str],
        calib_channels_dict: Optional[Dict[str, Sequence[float]]] = None,
    ) -> pd.DataFrame:
        """Calibrate all files with automatic model selection."""
        os.makedirs(self.config.output_folder, exist_ok=True)
        ref_masses = np.asarray(self.config.reference_masses, dtype=float)

        if calib_channels_dict is None:
            logger.info("Autodetecting calibrant channels...")
            calib_channels_dict = self._autodetect_channels(files, ref_masses)

        best_records: Dict[str, Dict] = {}
        max_workers = self.config.max_workers or os.cpu_count() or 4

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(
                    _fit_and_score_models_enhanced,
                    os.path.basename(fp), ref_masses,
                    calib_channels_dict, self.config,
                ): fp
                for fp in files
            }

            for fut in tqdm(as_completed(futs), total=len(futs), desc="Fitting models"):
                res = fut.result()
                if res is not None:
                    fname, rec = res
                    best_records[fname] = rec

        if not best_records:
            raise RuntimeError("No models could be fitted for any files")

        # Create summary
        rows = []
        for fname, rec in best_records.items():
            row = {
                "file_name": fname,
                "best_model": rec["best_model"],
                "best_ppm": rec[rec["best_model"]]["ppm"],
                "n_calibrants": rec[rec["best_model"]]["n_calibrants"],
            }
            for model in self.config.models_to_try:
                if model in rec:
                    row[f"{model}_ppm"] = rec[model]["ppm"]
            rows.append(row)

        summary = pd.DataFrame(rows).sort_values("file_name")
        summary.to_csv(
            os.path.join(self.config.output_folder, "calibration_summary.tsv"),
            sep="\t", index=False,
        )

        # Apply calibration
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            args = ((fp, best_records, self.config.output_folder) for fp in files)
            for _ in tqdm(ex.map(_apply_model_to_file_enhanced, args),
                         total=len(files), desc="Applying calibration"):
                pass

        mean_ppm = summary["best_ppm"].mean()
        median_ppm = summary["best_ppm"].median()
        logger.info(f"Calibration complete: Mean PPM = {mean_ppm:.1f}, Median PPM = {median_ppm:.1f}")

        return summary

    def _autodetect_channels(
        self,
        files: Sequence[str],
        ref_masses: npt.NDArray[np.float64],
    ) -> Dict[str, Sequence[float]]:
        """Autodetect calibrant channels from files."""
        autodetected: Dict[str, Sequence[float]] = {}
        self.last_autodetect_methods = {}

        for fp in tqdm(files, desc="Autodetecting channels"):
            try:
                df = pd.read_csv(fp, sep="\t", header=0, comment="#")
            except Exception as e:
                logger.warning(f"Failed to read {os.path.basename(fp)}: {e}")
                continue

            fname = os.path.basename(fp)

            if "Channel" not in df.columns:
                logger.warning(f"{fname}: No Channel column")
                continue

            if (self.config.prefer_recompute_from_channel
                    or self.config.autodetect_strategy == "bootstrap"
                    or "m/z" not in df.columns):
                ch = df["Channel"].to_numpy()
                y = df["Intensity"].astype("float64").to_numpy()
                autodetected[fname] = _enhanced_bootstrap_channels(ch, y, ref_masses)
                self.last_autodetect_methods[fname] = ["bootstrap"] * len(ref_masses)
            else:
                channels, methods_used = _enhanced_pick_channels(
                    df, ref_masses,
                    self.config.autodetect_tol_da,
                    self.config.autodetect_tol_ppm,
                    self.config.autodetect_method,
                    fallback_policy=self.config.autodetect_fallback_policy,
                    return_details=True,
                )
                autodetected[fname] = channels
                self.last_autodetect_methods[fname] = methods_used

                method_counts = pd.Series(methods_used).value_counts()
                non_requested = method_counts.drop(
                    labels=[self.config.autodetect_method, "none"],
                    errors="ignore",
                )
                if not non_requested.empty:
                    logger.info(
                        "%s: autodetect requested '%s' with fallback_policy='%s'; actual methods=%s",
                        fname,
                        self.config.autodetect_method,
                        self.config.autodetect_fallback_policy,
                        ", ".join(
                            f"{method_name} x{count}"
                            for method_name, count in non_requested.items()
                        ),
                    )

        if not autodetected:
            raise RuntimeError("Autodetection failed: no suitable files")

        logger.info(f"Autodetected channels for {len(autodetected)}/{len(files)} files")

        return autodetected


# ---------------------------------------------------------------------------
#  Convenience functions
# ---------------------------------------------------------------------------

def quick_calibrate(
    files: Sequence[str],
    reference_masses: Optional[List[float]] = None,
    output_folder: str = "calibrated_spectra",
    models: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Quick calibration with sensible defaults."""
    if reference_masses is None:
        reference_masses = [1.00782503224, 22.9897692820, 38.9637064864]

    if models is None:
        models = ["quad_sqrt", "reflectron", "linear_sqrt", "poly2"]

    config = AutoCalibConfig(
        reference_masses=reference_masses,
        output_folder=output_folder,
        models_to_try=models,
        **kwargs,
    )

    calibrator = AutoCalibrator(config)
    return calibrator.calibrate(files)


def diagnose_calibration(
    files: Sequence[str],
    reference_masses: List[float],
    output_folder: str = "calibration_diagnostics",
    calib_channels_dict: Optional[Dict[str, Sequence[float]]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run comprehensive calibration diagnostics.

    Tests multiple models and peak detection methods to find optimal settings.
    """
    results = {}
    os.makedirs(output_folder, exist_ok=True)

    all_models = ["quad_sqrt", "reflectron", "linear_sqrt", "poly2", "spline"]
    model_results = []

    logger.info("=" * 60)
    logger.info("CALIBRATION DIAGNOSTICS")
    logger.info("=" * 60)

    for models_to_test in [
        ["quad_sqrt"],
        ["reflectron"],
        ["linear_sqrt"],
        ["poly2"],
        ["spline"],
        ["quad_sqrt", "reflectron"],
        all_models,
    ]:
        logger.info(f"Testing models: {', '.join(models_to_test)}")

        config = AutoCalibConfig(
            reference_masses=reference_masses,
            output_folder=os.path.join(output_folder, "_".join(models_to_test)),
            models_to_try=models_to_test,
            use_outlier_rejection=True,
        )

        try:
            calibrator = AutoCalibrator(config)
            summary = calibrator.calibrate(files[:min(5, len(files))], calib_channels_dict)

            model_results.append({
                "models": ", ".join(models_to_test),
                "mean_ppm": summary["best_ppm"].mean(),
                "median_ppm": summary["best_ppm"].median(),
                "max_ppm": summary["best_ppm"].max(),
                "files_processed": len(summary),
            })

            logger.info(f"  Mean PPM: {summary['best_ppm'].mean():.1f}")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            model_results.append({
                "models": ", ".join(models_to_test),
                "mean_ppm": np.nan,
                "median_ppm": np.nan,
                "max_ppm": np.nan,
                "files_processed": 0,
            })

    results['model_comparison'] = pd.DataFrame(model_results)

    # Test different peak detection methods
    peak_methods = ["max", "centroid", "parabolic", "gaussian", "voigt"]
    peak_results = []

    logger.info("-" * 60)
    logger.info("Testing peak detection methods...")
    logger.info("-" * 60)

    for method in peak_methods:
        logger.info(f"Testing {method} peak detection")

        config = AutoCalibConfig(
            reference_masses=reference_masses,
            output_folder=os.path.join(output_folder, f"peak_{method}"),
            models_to_try=["quad_sqrt"],
            autodetect_method=method,
        )

        try:
            calibrator = AutoCalibrator(config)
            summary = calibrator.calibrate(files[:min(5, len(files))])

            peak_results.append({
                "method": method,
                "mean_ppm": summary["best_ppm"].mean(),
                "median_ppm": summary["best_ppm"].median(),
                "files_processed": len(summary),
            })

            logger.info(f"  Mean PPM: {summary['best_ppm'].mean():.1f}")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            peak_results.append({
                "method": method,
                "mean_ppm": np.nan,
                "median_ppm": np.nan,
                "files_processed": 0,
            })

    results['peak_method_comparison'] = pd.DataFrame(peak_results)

    for name, df in results.items():
        df.to_csv(os.path.join(output_folder, f"{name}.tsv"), sep="\t", index=False)

    logger.info("=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)

    if not results['model_comparison'].empty:
        best_model_idx = results['model_comparison']['mean_ppm'].idxmin()
        if not pd.isna(best_model_idx):
            best_models = results['model_comparison'].loc[best_model_idx, 'models']
            best_ppm = results['model_comparison'].loc[best_model_idx, 'mean_ppm']
            logger.info("Best model configuration: %s", best_models)
            logger.info("  Mean PPM error: %.1f", best_ppm)

    if not results['peak_method_comparison'].empty:
        best_peak_idx = results['peak_method_comparison']['mean_ppm'].idxmin()
        if not pd.isna(best_peak_idx):
            best_peak = results['peak_method_comparison'].loc[best_peak_idx, 'method']
            best_peak_ppm = results['peak_method_comparison'].loc[best_peak_idx, 'mean_ppm']
            logger.info("Best peak detection method: %s", best_peak)
            logger.info("  Mean PPM error: %.1f", best_peak_ppm)

    logger.info("Diagnostic results saved to: %s", output_folder)
    logger.info("=" * 60)

    return results


def validate_calibration(
    original_files: Sequence[str],
    calibrated_files: Sequence[str],
    known_masses: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Validate calibration quality by checking known masses."""
    if known_masses is None:
        known_masses = [
            1.0072764666,   # H+
            15.0229265168,  # CH3+
            22.9892207021,  # Na+
            27.0229265168,  # C2H3+
            29.0385765812,  # C2H5+
            38.9631579065,  # K+
            41.0385765812,  # C3H5+
            43.0542266457,  # C3H7+
            57.0698767102,  # C4H9+
            58.065674,      # C3H8N+
            67.0548,        # C5H7+
            71.0855267746,  # C5H11+
            86.096976,      # C5H12N+
            91.0542266457,  # C7H7+ (tropylium)
            104.107539,     # C5H14NO+
            184.073320,     # C5H15NO4P+ (phosphocholine)
            224.105171,     # C8H19NO4P+
            369.351600,     # C27H45+ (cholesterol)
        ]

    validation_results = []

    for orig_file, cal_file in zip(original_files, calibrated_files):
        try:
            cal_df = pd.read_csv(cal_file, sep="\t", comment="#")
            mz = cal_df["m/z"].to_numpy()
            intensity = cal_df["Intensity"].to_numpy()

            for known_mz in known_masses:
                mask = np.abs(mz - known_mz) < 0.5
                if mask.any():
                    idx_max = np.argmax(intensity[mask])
                    found_mz = mz[mask][idx_max]
                    found_int = intensity[mask][idx_max]
                    error_ppm = (found_mz - known_mz) / known_mz * 1e6

                    validation_results.append({
                        "file": os.path.basename(cal_file),
                        "known_mz": known_mz,
                        "found_mz": found_mz,
                        "intensity": found_int,
                        "error_ppm": error_ppm,
                    })

        except Exception as e:
            logger.warning(f"Failed to validate {cal_file}: {e}")

    if not validation_results:
        return pd.DataFrame()

    df = pd.DataFrame(validation_results)

    logger.info("=" * 60)
    logger.info("CALIBRATION VALIDATION SUMMARY")
    logger.info("=" * 60)

    for known_mz in known_masses:
        mz_data = df[df["known_mz"] == known_mz]["error_ppm"]
        if not mz_data.empty:
            mean_error = mz_data.mean()
            std_error = mz_data.std()
            logger.info("m/z %7.3f: %+6.1f +/- %5.1f ppm", known_mz, mean_error, std_error)

    overall_mean = df["error_ppm"].abs().mean()
    overall_median = df["error_ppm"].abs().median()
    logger.info("Overall mean absolute error: %.1f ppm", overall_mean)
    logger.info("Overall median absolute error: %.1f ppm", overall_median)
    logger.info("=" * 60)

    return df


def batch_process_directory(
    input_dir: str,
    output_dir: str = "calibrated_output",
    pattern: str = "*.txt",
    reference_masses: Optional[List[float]] = None,
    config: Optional[AutoCalibConfig] = None,
    recursive: bool = False,
) -> pd.DataFrame:
    """Process all matching files in a directory."""
    import glob

    if recursive:
        files = glob.glob(os.path.join(input_dir, "**", pattern), recursive=True)
    else:
        files = glob.glob(os.path.join(input_dir, pattern))

    if not files:
        raise ValueError(f"No files matching '{pattern}' found in {input_dir}")

    logger.info("Found %d files to process", len(files))

    if config is None:
        if reference_masses is None:
            reference_masses = [1.00782503224, 22.9897692820, 38.9637064864]

        config = AutoCalibConfig(
            reference_masses=reference_masses,
            output_folder=output_dir,
        )

    calibrator = AutoCalibrator(config)
    return calibrator.calibrate(files)


def create_calibration_report(
    summary_df: pd.DataFrame,
    output_file: str = "calibration_report.html",
) -> None:
    """Create an HTML report from calibration results."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Calibration Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .good {{ color: green; font-weight: bold; }}
            .warning {{ color: orange; font-weight: bold; }}
            .bad {{ color: red; font-weight: bold; }}
            .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Mass Spectrometry Calibration Report</h1>
        <div class="summary">
            <h2>Summary Statistics</h2>
            <p><strong>Files Processed:</strong> {n_files}</p>
            <p><strong>Mean PPM Error:</strong> <span class="{mean_class}">{mean_ppm:.1f}</span></p>
            <p><strong>Median PPM Error:</strong> <span class="{median_class}">{median_ppm:.1f}</span></p>
            <p><strong>Max PPM Error:</strong> <span class="{max_class}">{max_ppm:.1f}</span></p>
        </div>

        <h2>Model Usage</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Files</th>
                <th>Percentage</th>
            </tr>
            {model_rows}
        </table>

        <h2>File Details</h2>
        <table>
            <tr>
                <th>File</th>
                <th>Best Model</th>
                <th>PPM Error</th>
                <th>Calibrants</th>
            </tr>
            {file_rows}
        </table>

        <p><small>Report generated: {timestamp}</small></p>
    </body>
    </html>
    """

    mean_ppm = summary_df["best_ppm"].mean()
    median_ppm = summary_df["best_ppm"].median()
    max_ppm = summary_df["best_ppm"].max()

    def get_class(value):
        if value < 50:
            return "good"
        elif value < 100:
            return "warning"
        else:
            return "bad"

    model_counts = summary_df["best_model"].value_counts()
    model_rows = ""
    for model, count in model_counts.items():
        pct = count / len(summary_df) * 100
        model_rows += f"<tr><td>{model}</td><td>{count}</td><td>{pct:.1f}%</td></tr>\n"

    file_rows = ""
    for _, row in summary_df.iterrows():
        ppm_class = get_class(row["best_ppm"])
        file_rows += (
            f'<tr><td>{row["file_name"]}</td><td>{row["best_model"]}</td>'
            f'<td class="{ppm_class}">{row["best_ppm"]:.1f}</td>'
            f'<td>{row["n_calibrants"]}</td></tr>\n'
        )

    from datetime import datetime
    html = html_content.format(
        n_files=len(summary_df),
        mean_ppm=mean_ppm,
        mean_class=get_class(mean_ppm),
        median_ppm=median_ppm,
        median_class=get_class(median_ppm),
        max_ppm=max_ppm,
        max_class=get_class(max_ppm),
        model_rows=model_rows,
        file_rows=file_rows,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    with open(output_file, 'w') as f:
        f.write(html)

    logger.info("Report saved to: %s", output_file)


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Universal Mass Calibrator")
    parser.add_argument("files", nargs="+", help="Input files to calibrate")
    parser.add_argument("-o", "--output", default="calibrated_spectra",
                       help="Output directory")
    parser.add_argument("-r", "--reference", nargs="+", type=float,
                       help="Reference masses for calibration")
    parser.add_argument("-m", "--models", nargs="+",
                       choices=["quad_sqrt", "reflectron", "linear_sqrt", "poly2", "spline"],
                       help="Models to try")
    parser.add_argument("--diagnose", action="store_true",
                       help="Run diagnostic mode")
    parser.add_argument("--report", action="store_true",
                       help="Generate HTML report")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.diagnose:
        ref_masses = args.reference or [1.00782503224, 22.9897692820, 38.9637064864]
        diagnostics = diagnose_calibration(args.files, ref_masses)
    else:
        config = AutoCalibConfig(
            reference_masses=args.reference or [1.00782503224, 22.9897692820, 38.9637064864],
            output_folder=args.output,
            models_to_try=args.models or ["quad_sqrt", "reflectron", "linear_sqrt", "poly2"],
        )

        calibrator = AutoCalibrator(config)
        summary = calibrator.calibrate(args.files)

        logger.info("Calibration complete!")
        logger.info("Mean PPM error: %.1f", summary['best_ppm'].mean())
        logger.info("Files processed: %d", len(summary))
