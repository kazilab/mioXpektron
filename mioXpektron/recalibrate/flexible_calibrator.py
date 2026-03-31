"""
FlexibleCalibrator — single-model calibration with user-selected method.

Unlike ``AutoCalibrator`` which fits all models and picks the best,
``FlexibleCalibrator`` fits exactly *one* user-chosen model with optional
iterative outlier rejection and quality-control thresholds.

All model fitting, inversion, and peak-detection functions live in the
shared ``_models`` backend.

Author: Data Analysis Team @KaziLab.se
Version: 0.0.1
"""

import json
import os
import logging
import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- shared backend ---
from ._models import (
    MODEL_REGISTRY,
    _MODEL_ALIASES,
    _ppm_error,
    _ppm_to_da,
    _fit_gaussian_peak,
    _fit_voigt_peak,
    _parabolic_peak_center,
    _enhanced_pick_channels,
    _enhanced_bootstrap_channels,
    _fit_quad_sqrt_robust,
    _fit_reflectron,
    _fit_multisegment,
    _fit_spline_model,
    _fit_physical_tof,
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

# Type alias for calibration methods
CalibrationMethod = Literal[
    "quad_sqrt", "linear_sqrt", "poly2", "reflectron",
    "multisegment", "spline", "physical",
]


def _format_segment_metrics(segment_metrics: Sequence[Dict[str, object]]) -> str:
    """Render multisegment ppm diagnostics in a compact log-friendly format."""
    formatted: List[str] = []
    for metric in segment_metrics:
        upper = "inf" if metric["mz_max"] is None else f"{metric['mz_max']:.1f}"
        if metric.get("ppm_error") is None:
            value = str(metric.get("status", "unavailable"))
        else:
            value = f"{metric['ppm_error']:.1f}ppm"
        formatted.append(
            f"{metric['segment']}[{metric['mz_min']:.1f},{upper})="
            f"{value}(n={metric['n_calibrants']})"
        )
    return "; ".join(formatted)


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class FlexibleCalibConfig:
    """Configuration for flexible calibration with a single user-selected method."""

    reference_masses: List[float]
    calibration_method: CalibrationMethod = "quad_sqrt"
    output_folder: str = "calibrated_spectra"
    output_mz_range: Optional[Tuple[Optional[float], Optional[float]]] = None
    max_workers: Optional[int] = None

    # Peak detection parameters
    autodetect_tol_da: Optional[float | Sequence[float]] = None
    autodetect_tol_ppm: Optional[float] = None
    autodetect_method: str = "gaussian"
    autodetect_fallback_policy: str = "max"
    autodetect_strategy: str = "mz"
    prefer_recompute_from_channel: bool = False

    # Robust fitting parameters
    outlier_threshold: float = 3.0
    use_outlier_rejection: bool = True
    max_iterations: int = 3

    # Quality control
    min_calibrants: int = 3
    max_ppm_threshold: Optional[float] = 100.0
    fail_on_high_error: bool = False
    retry_high_error_with_pruning: bool = False
    retry_high_error_with_mz_fallback: bool = False
    retry_high_error_max_removals: int = 5
    exclude_reference_masses: List[float] = field(default_factory=list)
    auto_screen_reference_masses: bool = False
    screen_max_mean_abs_ppm: float = 50.0
    screen_max_median_abs_ppm: Optional[float] = None
    screen_min_valid_fraction: float = 0.8
    screen_min_count: int = 3
    screen_exclude_below_mz: float = 1.5

    # Advanced parameters for specific models
    spline_smoothing: Optional[float] = None
    multisegment_breakpoints: List[float] = field(default_factory=lambda: [50, 200, 500])
    instrument_params: Dict[str, float] = field(default_factory=dict)

    # Reporting options
    save_diagnostic_plots: bool = False
    verbose: bool = True

    # Adaptive parameterization (opt-in)
    auto_tune: bool = False

    def __post_init__(self):
        if self.autodetect_fallback_policy not in {"max", "nan", "raise"}:
            raise ValueError(
                "autodetect_fallback_policy must be 'max', 'nan', or 'raise'"
            )
        if self.output_mz_range is not None:
            if len(self.output_mz_range) != 2:
                raise ValueError("output_mz_range must be None or a (min_mz, max_mz) pair")
            mz_min, mz_max = self.output_mz_range
            if (
                mz_min is not None
                and mz_max is not None
                and mz_min > mz_max
            ):
                raise ValueError("output_mz_range must satisfy min_mz <= max_mz")
        if self.screen_max_mean_abs_ppm <= 0:
            raise ValueError("screen_max_mean_abs_ppm must be > 0")
        if self.screen_max_median_abs_ppm is not None and self.screen_max_median_abs_ppm <= 0:
            raise ValueError("screen_max_median_abs_ppm must be > 0 when provided")
        if not 0 < self.screen_min_valid_fraction <= 1:
            raise ValueError("screen_min_valid_fraction must be in (0, 1]")
        if self.screen_min_count < 1:
            raise ValueError("screen_min_count must be >= 1")
        if self.screen_exclude_below_mz < 0:
            raise ValueError("screen_exclude_below_mz must be >= 0")
        if self.retry_high_error_max_removals < 1:
            raise ValueError("retry_high_error_max_removals must be >= 1")


def _mask_excluded_reference_masses(
    reference_masses: npt.NDArray[np.float64],
    excluded_reference_masses: Sequence[float],
    *,
    atol: float = 1e-6,
) -> npt.NDArray[np.bool_]:
    """Return a mask that excludes user-specified reference masses."""
    ref = np.asarray(reference_masses, dtype=float)
    keep = np.ones(len(ref), dtype=bool)
    if not excluded_reference_masses:
        return keep

    excluded = np.asarray(excluded_reference_masses, dtype=float)
    for mass in excluded:
        keep &= ~np.isclose(ref, mass, atol=atol, rtol=0.0)
    return keep


def _filter_reference_masses(
    reference_masses: Sequence[float],
    excluded_reference_masses: Sequence[float],
) -> npt.NDArray[np.float64]:
    """Apply explicit reference-mass exclusions while preserving order."""
    ref = np.asarray(reference_masses, dtype=float)
    keep = _mask_excluded_reference_masses(ref, excluded_reference_masses)
    return ref[keep]


def _filter_calibration_values(
    calibration_values: Dict[str, Sequence[float]],
    original_reference_masses: Sequence[float],
    selected_reference_masses: Sequence[float],
) -> Dict[str, npt.NDArray[np.float64]]:
    """Subset per-file calibration values to a selected reference-mass list."""
    original = np.asarray(original_reference_masses, dtype=float)
    selected = np.asarray(selected_reference_masses, dtype=float)
    keep = np.zeros(len(original), dtype=bool)
    for mass in selected:
        keep |= np.isclose(original, mass, atol=1e-6, rtol=0.0)

    filtered: Dict[str, npt.NDArray[np.float64]] = {}
    for file_name, values in calibration_values.items():
        arr = np.asarray(values, dtype=float)
        if len(arr) != len(original):
            raise ValueError(
                f"{file_name}: calibration values length {len(arr)} does not match "
                f"{len(original)} reference masses"
            )
        filtered[file_name] = arr[keep]
    return filtered


def _filter_autodetect_methods(
    method_map: Dict[str, List[str]],
    original_reference_masses: Sequence[float],
    selected_reference_masses: Sequence[float],
) -> Dict[str, List[str]]:
    """Subset per-file autodetect method diagnostics to selected masses."""
    original = np.asarray(original_reference_masses, dtype=float)
    selected = np.asarray(selected_reference_masses, dtype=float)
    keep = np.zeros(len(original), dtype=bool)
    for mass in selected:
        keep |= np.isclose(original, mass, atol=1e-6, rtol=0.0)

    filtered: Dict[str, List[str]] = {}
    for file_name, methods in method_map.items():
        if len(methods) != len(original):
            filtered[file_name] = list(methods)
            continue
        filtered[file_name] = [method for method, use in zip(methods, keep) if use]
    return filtered


def summarize_reference_mass_stability(
    summary_df: pd.DataFrame,
    *,
    exclude_below_mz: float = 1.5,
) -> pd.DataFrame:
    """Summarize per-mass residual stability across the current calibration summary."""
    rows: List[Dict[str, float]] = []
    n_files = len(summary_df)
    if n_files == 0:
        return pd.DataFrame(
            columns=[
                "mass",
                "count_valid",
                "valid_fraction",
                "mean_abs_ppm",
                "median_abs_ppm",
                "std_ppm",
                "max_abs_ppm",
            ]
        )

    for _, row in summary_df.iterrows():
        masses = np.asarray(row.get("calibrant_masses", []), dtype=float)
        estimated = np.asarray(row.get("estimated_masses", []), dtype=float)
        if len(masses) != len(estimated):
            continue
        for mass, est in zip(masses, estimated):
            if not np.isfinite(mass) or mass <= exclude_below_mz:
                continue
            ppm_error = np.nan
            if np.isfinite(est) and est > 0 and mass > 0:
                ppm_error = float((est - mass) / mass * 1e6)
            rows.append(
                {
                    "file_name": row.get("file_name"),
                    "mass": float(mass),
                    "ppm_error": ppm_error,
                    "abs_ppm": abs(ppm_error) if np.isfinite(ppm_error) else np.nan,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "mass",
                "count_valid",
                "valid_fraction",
                "mean_abs_ppm",
                "median_abs_ppm",
                "std_ppm",
                "max_abs_ppm",
            ]
        )

    detail_df = pd.DataFrame(rows)
    grouped = (
        detail_df.groupby("mass", as_index=False)
        .agg(
            count_valid=("abs_ppm", lambda s: int(np.isfinite(s).sum())),
            mean_abs_ppm=("abs_ppm", lambda s: float(np.nanmean(s)) if np.isfinite(s).any() else np.nan),
            median_abs_ppm=("abs_ppm", lambda s: float(np.nanmedian(s)) if np.isfinite(s).any() else np.nan),
            std_ppm=("ppm_error", lambda s: float(np.nanstd(s)) if np.isfinite(s).any() else np.nan),
            max_abs_ppm=("abs_ppm", lambda s: float(np.nanmax(s)) if np.isfinite(s).any() else np.nan),
        )
        .sort_values("mass")
    )
    grouped["valid_fraction"] = grouped["count_valid"] / max(n_files, 1)
    return grouped[
        [
            "mass",
            "count_valid",
            "valid_fraction",
            "mean_abs_ppm",
            "median_abs_ppm",
            "std_ppm",
            "max_abs_ppm",
        ]
    ]


def _absolute_ppm_errors(
    true_masses: Sequence[float],
    estimated_masses: Sequence[float],
) -> npt.NDArray[np.float64]:
    """Return per-mass absolute ppm errors, preserving NaN for invalid entries."""
    true_arr = np.asarray(true_masses, dtype=float)
    est_arr = np.asarray(estimated_masses, dtype=float)
    abs_ppm = np.full_like(true_arr, np.nan, dtype=np.float64)
    valid = np.isfinite(true_arr) & np.isfinite(est_arr) & (true_arr > 0.0) & (est_arr > 0.0)
    if np.any(valid):
        abs_ppm[valid] = np.abs((est_arr[valid] - true_arr[valid]) / true_arr[valid] * 1e6)
    return abs_ppm


def _select_worst_calibrant_index(
    masses: Sequence[float],
    abs_ppm: Sequence[float],
    *,
    exclude_below_mz: float,
) -> Optional[int]:
    """Pick the worst calibrant to prune, preferring masses above the exclusion floor."""
    masses_arr = np.asarray(masses, dtype=float)
    ppm_arr = np.asarray(abs_ppm, dtype=float)
    valid = np.isfinite(masses_arr) & np.isfinite(ppm_arr)
    if not np.any(valid):
        return None

    preferred_idx = np.flatnonzero(valid & (masses_arr > exclude_below_mz))
    if preferred_idx.size > 0:
        local = preferred_idx[np.argmax(ppm_arr[preferred_idx])]
        return int(local)

    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return None
    local = valid_idx[np.argmax(ppm_arr[valid_idx])]
    return int(local)


def select_stable_reference_masses(
    reference_masses: Sequence[float],
    stability_df: pd.DataFrame,
    *,
    exclude_below_mz: float = 1.5,
    max_mean_abs_ppm: float = 50.0,
    max_median_abs_ppm: Optional[float] = None,
    min_valid_fraction: float = 0.8,
    min_count: int = 3,
) -> Tuple[npt.NDArray[np.float64], pd.DataFrame]:
    """Select a stable reference-mass subset based on cross-spectrum residuals."""
    ref = np.asarray(reference_masses, dtype=float)
    annotated = stability_df.copy()
    if annotated.empty:
        return ref.copy(), annotated

    annotated["keep_by_count"] = annotated["count_valid"] >= min_count
    annotated["keep_by_fraction"] = annotated["valid_fraction"] >= min_valid_fraction
    annotated["keep_by_mean"] = annotated["mean_abs_ppm"] <= max_mean_abs_ppm
    if max_median_abs_ppm is None:
        annotated["keep_by_median"] = True
    else:
        annotated["keep_by_median"] = annotated["median_abs_ppm"] <= max_median_abs_ppm
    annotated["selected"] = (
        annotated["keep_by_count"]
        & annotated["keep_by_fraction"]
        & annotated["keep_by_mean"]
        & annotated["keep_by_median"]
    )

    keep = []
    for mass in ref:
        if mass <= exclude_below_mz:
            keep.append(True)
            continue
        match = annotated[np.isclose(annotated["mass"], mass, atol=1e-6, rtol=0.0)]
        if match.empty:
            keep.append(False)
        else:
            keep.append(bool(match.iloc[0]["selected"]))

    return ref[np.asarray(keep, dtype=bool)], annotated


# ---------------------------------------------------------------------------
#  Fit a single selected model to one file
# ---------------------------------------------------------------------------

def _fit_selected_model_enhanced(
    filename: str,
    ref_masses: npt.NDArray[np.float64],
    calib_channels_dict: Dict[str, Sequence[float]],
    config: FlexibleCalibConfig,
) -> Optional[Tuple[str, Dict]]:
    """Fit the user-selected calibration model to *filename*."""
    if filename not in calib_channels_dict:
        logger.warning(f"{filename}: Not found in calibration channels dict")
        return None

    t_meas = np.asarray(calib_channels_dict[filename], dtype=float)
    m_ref = np.asarray(ref_masses, dtype=float)

    valid = np.isfinite(t_meas)
    n_valid = valid.sum()

    min_required = 2 if config.calibration_method == "linear_sqrt" else 3
    if n_valid < min_required:
        logger.warning(f"{filename}: Only {n_valid} valid calibrants (need >={min_required})")
        return None

    t_meas = t_meas[valid]
    m_ref = m_ref[valid]

    logger.debug(f"{filename}: Fitting {config.calibration_method} with {n_valid} calibrants")

    params = None
    segment_metrics: List[Dict[str, object]] = []
    method = config.calibration_method

    if method == "quad_sqrt":
        if config.use_outlier_rejection:
            params = _fit_quad_sqrt_robust(
                m_ref, t_meas,
                config.outlier_threshold,
                config.max_iterations,
            )
        else:
            params = _fit_quad_sqrt_robust(m_ref, t_meas, max_iterations=1)

        if params is None:
            logger.error(f"{filename}: TOF model fitting failed")
            return None
        m_est = _invert_quad_sqrt(t_meas, *params)

    elif method == "reflectron":
        params = _fit_reflectron(m_ref, t_meas)
        if params is None:
            logger.error(f"{filename}: Reflectron model fitting failed")
            return None
        m_est = _invert_reflectron(t_meas, *params)

    elif method == "multisegment":
        params = _fit_multisegment(m_ref, t_meas, config.multisegment_breakpoints)
        if params is None:
            logger.error(f"{filename}: Multisegment model fitting failed")
            return None
        # Evaluate ppm by applying each segment to its calibrants
        m_est = np.full_like(m_ref, np.nan)
        all_breakpoints = [0.0] + [float(bp) for bp in sorted(config.multisegment_breakpoints)] + [np.inf]
        for i in range(len(all_breakpoints) - 1):
            low, high = all_breakpoints[i], all_breakpoints[i + 1]
            seg_mask = (m_ref >= low) & (m_ref < high)
            if not seg_mask.any():
                continue

            segment_name = f"segment_{i}"
            seg_info = params["segments"].get(segment_name)
            metric: Dict[str, object] = {
                "segment": segment_name,
                "mz_min": float(low),
                "mz_max": None if np.isinf(high) else float(high),
                "n_calibrants": int(seg_mask.sum()),
            }

            if seg_info is not None:
                seg_m_est = _invert_quad_sqrt(t_meas[seg_mask], *seg_info["params"])
                m_est[seg_mask] = seg_m_est
                metric["ppm_error"] = float(_ppm_error(m_ref[seg_mask], seg_m_est))
                metric["status"] = "fitted"
            elif int(seg_mask.sum()) < 3:
                metric["ppm_error"] = None
                metric["status"] = "insufficient_calibrants"
            else:
                metric["ppm_error"] = None
                metric["status"] = "fit_failed"

            segment_metrics.append(metric)

    elif method == "spline":
        params = _fit_spline_model(m_ref, t_meas, config.spline_smoothing)
        if params is None:
            logger.error(f"{filename}: Spline model fitting failed")
            return None
        m_est = _invert_spline(t_meas, params)

    elif method == "physical":
        if not config.instrument_params:
            logger.error(f"{filename}: Physical model requires instrument parameters")
            return None
        params = _fit_physical_tof(m_ref, t_meas, config.instrument_params)
        if params is None:
            logger.error(f"{filename}: Physical model fitting failed")
            return None
        L = config.instrument_params['flight_length']
        V = config.instrument_params['acceleration_voltage']
        m_est = _invert_physical(t_meas, *params, L, V)

    elif method == "linear_sqrt":
        params = _fit_linear_sqrt(m_ref, t_meas)
        if params is None:
            logger.error(f"{filename}: Linear sqrt model fitting failed")
            return None
        m_est = _invert_linear_sqrt(t_meas, *params)

    elif method == "poly2":
        params = _fit_poly2(m_ref, t_meas)
        if params is None:
            logger.error(f"{filename}: Poly2 model fitting failed")
            return None
        m_est = _invert_poly2(t_meas, *params)

    else:
        logger.error(f"{filename}: Unknown calibration method '{method}'")
        return None

    ppm = _ppm_error(m_ref, m_est)

    if (
        config.fail_on_high_error
        and config.max_ppm_threshold
        and not config.retry_high_error_with_pruning
    ):
        if ppm > config.max_ppm_threshold:
            logger.error(f"{filename}: PPM error {ppm:.1f} exceeds threshold {config.max_ppm_threshold}")
            return None

    result = {
        "method": method,
        "params": params,
        "ppm": ppm,
        "n_calibrants": n_valid,
        "calibrant_masses": m_ref.tolist(),
        "calibrant_channels": t_meas.tolist(),
        "estimated_masses": m_est.tolist(),
        "segment_metrics": segment_metrics,
        "rescue_initial_ppm": np.nan,
        "rescue_n_removed": 0,
        "rescue_removed_masses": [],
        "rescue_strategy": "",
    }

    logger.info(f"{filename}: {method.upper()} fitted, ppm={ppm:.1f}, n={n_valid}")
    if method == "multisegment" and segment_metrics:
        logger.info(
            "%s: multisegment segment ppm: %s",
            filename,
            _format_segment_metrics(segment_metrics),
        )

    return filename, result


# ---------------------------------------------------------------------------
#  Apply calibration to a single file
# ---------------------------------------------------------------------------

def _apply_model_to_file_enhanced(
    args: Tuple[
        str,
        Dict[str, Dict],
        str,
        CalibrationMethod,
        Dict[str, float],
        Optional[Tuple[Optional[float], Optional[float]]],
    ],
) -> Optional[str]:
    """Apply the fitted calibration model to a file."""
    file_path, results_map, out_folder, method, instrument_params, output_mz_range = args
    fname = os.path.basename(file_path)

    if fname not in results_map:
        logger.warning(f"{fname}: No calibration results found")
        return None

    rec = results_map[fname]
    params = rec["params"]

    try:
        df = pd.read_csv(file_path, sep="\t", header=0, comment="#")
    except Exception as e:
        logger.error(f"{fname}: Failed to read file - {e}")
        return None

    if "Channel" not in df.columns:
        logger.error(f"{fname}: Missing 'Channel' column")
        return None

    t = df["Channel"].astype(np.float64).to_numpy()

    mz_cal = apply_model_to_spectrum(t, method, params, instrument_params)

    out_df = pd.DataFrame({
        "Channel": t,
        "m/z": mz_cal,
        "Intensity": df["Intensity"].to_numpy(),
    })

    if output_mz_range is not None:
        mz_min, mz_max = output_mz_range
        keep = np.ones(len(out_df), dtype=bool)
        if mz_min is not None:
            keep &= out_df["m/z"].to_numpy() >= mz_min
        if mz_max is not None:
            keep &= out_df["m/z"].to_numpy() <= mz_max
        out_df = out_df.loc[keep].reset_index(drop=True)

    os.makedirs(out_folder, exist_ok=True)
    out_fp = os.path.join(out_folder, fname.replace(".txt", "_calibrated.txt"))

    with open(out_fp, 'w') as f:
        f.write(f"# Calibration method: {method}\n")
        f.write(f"# Calibration error: {rec['ppm']:.1f} ppm\n")
        f.write(f"# Number of calibrants: {rec['n_calibrants']}\n")
        if rec.get("segment_metrics"):
            f.write(f"# Segment metrics: {json.dumps(rec['segment_metrics'])}\n")
        if rec.get("rescue_strategy"):
            f.write(f"# Rescue strategy: {rec['rescue_strategy']}\n")
        if rec.get("rescue_n_removed", 0):
            f.write(f"# Rescue initial ppm: {rec['rescue_initial_ppm']:.1f}\n")
            f.write(f"# Rescue removed masses: {rec['rescue_removed_masses']}\n")
        f.write(f"# Calibrant masses: {rec['calibrant_masses']}\n")
        if output_mz_range is not None:
            f.write(f"# Saved m/z range: {list(output_mz_range)}\n")
        out_df.to_csv(f, sep="\t", index=False)

    logger.debug(f"{fname}: Saved calibrated spectrum to {out_fp}")

    return fname


# ---------------------------------------------------------------------------
#  Main calibrator class
# ---------------------------------------------------------------------------

class FlexibleCalibrator:
    """Single-model calibrator with user-selected method.

    Unlike ``AutoCalibrator``, this calibrator fits exactly one model and
    provides more control over outlier rejection, quality thresholds, and
    per-model parameters.
    """

    def __init__(self, config: Optional[FlexibleCalibConfig] = None):
        default_masses = [
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

        self.config = config or FlexibleCalibConfig(
            reference_masses=default_masses,
            calibration_method="quad_sqrt",
        )
        self.last_autodetect_methods: Dict[str, List[str]] = {}
        self.last_reference_masses_initial: List[float] = []
        self.last_reference_masses_used: List[float] = []
        self.last_reference_masses_screened_out: List[float] = []
        self.last_reference_mass_screening: pd.DataFrame = pd.DataFrame()
        self.last_failed_or_excluded_files: pd.DataFrame = pd.DataFrame()

        logger.info(f"FlexibleCalibrator initialized with method={self.config.calibration_method}")

    def _fit_results_map(
        self,
        files: Sequence[str],
        calib_channels_dict: Dict[str, Sequence[float]],
        ref_masses: npt.NDArray[np.float64],
    ) -> Dict[str, Dict]:
        """Fit the selected model for all files and return a results map."""
        method = self.config.calibration_method
        results_map: Dict[str, Dict] = {}
        max_workers = self.config.max_workers or os.cpu_count() or 4

        if max_workers == 1:
            for fp in tqdm(files, desc=f"Fitting {method} model"):
                res = _fit_selected_model_enhanced(
                    os.path.basename(fp), ref_masses,
                    calib_channels_dict, self.config,
                )
                if res is not None:
                    fname, rec = res
                    results_map[fname] = rec
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(
                        _fit_selected_model_enhanced,
                        os.path.basename(fp), ref_masses,
                        calib_channels_dict, self.config,
                    ): fp
                    for fp in files
                }

                for fut in tqdm(as_completed(futs), total=len(futs),
                              desc=f"Fitting {method} model"):
                    res = fut.result()
                    if res is not None:
                        fname, rec = res
                        results_map[fname] = rec
        return results_map

    @staticmethod
    def _build_summary(results_map: Dict[str, Dict]) -> pd.DataFrame:
        """Convert a results map into the standard summary dataframe."""
        rows = []
        for fname, rec in results_map.items():
            rows.append({
                "file_name": fname,
                "method": rec["method"],
                "ppm_error": rec["ppm"],
                "n_calibrants": rec["n_calibrants"],
                "calibrant_masses": rec["calibrant_masses"],
                "calibrant_channels": rec["calibrant_channels"],
                "estimated_masses": rec["estimated_masses"],
                "segment_metrics": json.dumps(rec.get("segment_metrics", [])),
                "rescue_initial_ppm": rec.get("rescue_initial_ppm", np.nan),
                "rescue_n_removed": rec.get("rescue_n_removed", 0),
                "rescue_removed_masses": rec.get("rescue_removed_masses", []),
                "rescue_strategy": rec.get("rescue_strategy", ""),
            })
        return pd.DataFrame(rows).sort_values("file_name")

    @staticmethod
    def _build_issue_summary(
        *,
        missing_files: Sequence[str],
        results_map: Dict[str, Dict],
        excluded_high_error_files: Sequence[str],
        threshold_ppm: Optional[float],
    ) -> pd.DataFrame:
        """Summarize files that failed to fit or were excluded after fitting."""
        columns = [
            "file_name",
            "status",
            "reason",
            "ppm_error",
            "threshold_ppm",
            "n_calibrants",
            "rescue_strategy",
            "rescue_initial_ppm",
        ]
        rows = []

        for fname in sorted(set(missing_files)):
            rows.append({
                "file_name": fname,
                "status": "fit_failed",
                "reason": "model_fit_failed_or_insufficient_calibrants",
                "ppm_error": np.nan,
                "threshold_ppm": threshold_ppm,
                "n_calibrants": np.nan,
                "rescue_strategy": "",
                "rescue_initial_ppm": np.nan,
            })

        for fname in sorted(set(excluded_high_error_files)):
            rec = results_map.get(fname, {})
            ppm_error = float(rec.get("ppm", np.nan))
            if threshold_ppm is not None and np.isfinite(ppm_error):
                reason = f"ppm_error_above_threshold ({ppm_error:.2f} > {threshold_ppm:.2f})"
            else:
                reason = "ppm_error_above_threshold"
            rows.append({
                "file_name": fname,
                "status": "excluded_high_ppm",
                "reason": reason,
                "ppm_error": ppm_error,
                "threshold_ppm": threshold_ppm,
                "n_calibrants": rec.get("n_calibrants", np.nan),
                "rescue_strategy": rec.get("rescue_strategy", ""),
                "rescue_initial_ppm": rec.get("rescue_initial_ppm", np.nan),
            })

        if not rows:
            return pd.DataFrame(columns=columns)

        issue_df = pd.DataFrame(rows, columns=columns)
        return issue_df.sort_values(["status", "file_name"]).reset_index(drop=True)

    def _fit_single_file_result(
        self,
        filename: str,
        ref_masses: npt.NDArray[np.float64],
        channels: npt.NDArray[np.float64],
        config: FlexibleCalibConfig,
    ) -> Optional[Dict]:
        """Fit one file using a pre-selected subset of calibrant masses/channels."""
        result = _fit_selected_model_enhanced(
            filename,
            ref_masses,
            {filename: channels},
            config,
        )
        if result is None:
            return None
        _fname, rec = result
        return rec

    def _fit_single_file_with_strategy(
        self,
        file_path: str,
        ref_masses: npt.NDArray[np.float64],
        *,
        autodetect_strategy: str,
        prefer_recompute_from_channel: bool,
    ) -> Optional[Dict]:
        """Autodetect and fit one file sequentially with an alternate strategy."""
        filename = os.path.basename(file_path)
        rescue_config = replace(
            self.config,
            autodetect_strategy=autodetect_strategy,
            prefer_recompute_from_channel=prefer_recompute_from_channel,
            max_workers=1,
            fail_on_high_error=False,
            retry_high_error_with_pruning=False,
            retry_high_error_with_mz_fallback=False,
        )
        rescue_calibrator = FlexibleCalibrator(rescue_config)
        channels_map = rescue_calibrator._autodetect_channels([file_path], ref_masses)
        if filename not in channels_map:
            return None

        channels = np.asarray(channels_map[filename], dtype=float)
        valid = np.isfinite(channels)
        if valid.sum() < (2 if rescue_config.calibration_method == "linear_sqrt" else 3):
            return None
        return rescue_calibrator._fit_single_file_result(
            filename,
            np.asarray(ref_masses, dtype=float)[valid],
            channels[valid],
            rescue_config,
        )

    def _rescue_high_error_file(
        self,
        filename: str,
        file_path: str,
        ref_masses: npt.NDArray[np.float64],
        calib_channels_dict: Dict[str, Sequence[float]],
        initial_rec: Dict,
    ) -> Optional[Dict]:
        """Retry one high-error fit sequentially while pruning the worst calibrants."""
        if filename not in calib_channels_dict:
            return None

        threshold = self.config.max_ppm_threshold
        if threshold is None:
            return None

        channels_full = np.asarray(calib_channels_dict[filename], dtype=float)
        masses_full = np.asarray(ref_masses, dtype=float)
        valid = np.isfinite(channels_full) & np.isfinite(masses_full)
        work_channels = channels_full[valid].copy()
        work_masses = masses_full[valid].copy()

        min_required = 2 if self.config.calibration_method == "linear_sqrt" else 3
        max_removals = min(
            self.config.retry_high_error_max_removals,
            max(len(work_masses) - min_required, 0),
        )
        if max_removals <= 0:
            return None

        rescue_config = replace(
            self.config,
            max_workers=1,
            fail_on_high_error=False,
        )
        current_rec = initial_rec
        best_rec = initial_rec
        removed_masses: List[float] = []
        best_removed_masses: List[float] = []

        for _ in range(max_removals):
            abs_ppm = _absolute_ppm_errors(
                current_rec["calibrant_masses"],
                current_rec["estimated_masses"],
            )
            drop_idx = _select_worst_calibrant_index(
                current_rec["calibrant_masses"],
                abs_ppm,
                exclude_below_mz=self.config.screen_exclude_below_mz,
            )
            if drop_idx is None or len(work_masses) <= min_required:
                break

            removed_masses.append(float(work_masses[drop_idx]))
            work_masses = np.delete(work_masses, drop_idx)
            work_channels = np.delete(work_channels, drop_idx)

            rescued_rec = self._fit_single_file_result(
                filename,
                work_masses,
                work_channels,
                rescue_config,
            )
            if rescued_rec is None:
                break

            current_rec = rescued_rec
            if np.isfinite(current_rec["ppm"]) and current_rec["ppm"] < best_rec["ppm"]:
                best_rec = current_rec
                best_removed_masses = list(removed_masses)

            if np.isfinite(current_rec["ppm"]) and current_rec["ppm"] <= threshold:
                break

        rescue_candidate: Optional[Dict] = None
        if np.isfinite(best_rec["ppm"]) and best_rec["ppm"] < initial_rec["ppm"]:
            rescue_candidate = dict(best_rec)
            rescue_candidate["rescue_initial_ppm"] = float(initial_rec["ppm"])
            rescue_candidate["rescue_n_removed"] = len(best_removed_masses)
            rescue_candidate["rescue_removed_masses"] = best_removed_masses
            rescue_candidate["rescue_strategy"] = "prune_worst_calibrants"

        if (
            self.config.retry_high_error_with_mz_fallback
            and self.config.autodetect_strategy == "bootstrap"
            and (rescue_candidate is None or rescue_candidate["ppm"] > threshold)
        ):
            mz_rec = self._fit_single_file_with_strategy(
                file_path,
                ref_masses,
                autodetect_strategy="mz",
                prefer_recompute_from_channel=False,
            )
            if mz_rec is not None and np.isfinite(mz_rec["ppm"]):
                baseline_ppm = initial_rec["ppm"] if rescue_candidate is None else rescue_candidate["ppm"]
                if mz_rec["ppm"] < baseline_ppm:
                    rescue_candidate = dict(mz_rec)
                    rescue_candidate["rescue_initial_ppm"] = float(initial_rec["ppm"])
                    rescue_candidate["rescue_n_removed"] = 0
                    rescue_candidate["rescue_removed_masses"] = []
                    rescue_candidate["rescue_strategy"] = "mz_fallback"

        return rescue_candidate

    def _rescue_high_error_results_map(
        self,
        results_map: Dict[str, Dict],
        files: Sequence[str],
        calib_channels_dict: Dict[str, Sequence[float]],
        ref_masses: npt.NDArray[np.float64],
    ) -> Dict[str, Dict]:
        """Retry catastrophic fits sequentially before any threshold-based exclusion."""
        threshold = self.config.max_ppm_threshold
        if not self.config.retry_high_error_with_pruning or threshold is None:
            return results_map

        high_error_files = [
            fname for fname, rec in results_map.items()
            if not np.isfinite(rec["ppm"]) or rec["ppm"] > threshold
        ]
        if not high_error_files:
            return results_map

        logger.warning(
            "Sequential rescue pass for %d files above %.1f ppm",
            len(high_error_files),
            threshold,
        )

        file_map = {os.path.basename(fp): fp for fp in files}
        improved = 0
        rescued_to_threshold = 0
        for fname in high_error_files:
            initial_ppm = float(results_map[fname]["ppm"])
            file_path = file_map.get(fname)
            if file_path is None:
                continue
            rescued_rec = self._rescue_high_error_file(
                fname,
                file_path,
                ref_masses,
                calib_channels_dict,
                results_map[fname],
            )
            if rescued_rec is None:
                continue

            results_map[fname] = rescued_rec
            improved += 1
            if rescued_rec["ppm"] <= threshold:
                rescued_to_threshold += 1
            logger.warning(
                "%s: rescue improved ppm from %.1f to %.1f using %s",
                fname,
                initial_ppm,
                rescued_rec["ppm"],
                rescued_rec.get("rescue_strategy", "rescue"),
            )

        logger.info(
            "Rescue pass improved %d/%d high-error files; %d now pass the %.1f ppm threshold",
            improved,
            len(high_error_files),
            rescued_to_threshold,
            threshold,
        )
        return results_map

    def _apply_results_map(
        self,
        files: Sequence[str],
        results_map: Dict[str, Dict],
    ) -> int:
        """Apply fitted models to all files and return the number of successes."""
        method = self.config.calibration_method
        success_count = 0
        max_workers = self.config.max_workers or os.cpu_count() or 4

        if max_workers == 1:
            for fp in tqdm(files, desc=f"Applying {method} model"):
                result = _apply_model_to_file_enhanced(
                    (fp, results_map, self.config.output_folder, method,
                     self.config.instrument_params, self.config.output_mz_range),
                )
                if result is not None:
                    success_count += 1
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                args = (
                    (fp, results_map, self.config.output_folder, method,
                     self.config.instrument_params, self.config.output_mz_range)
                    for fp in files
                )
                for result in tqdm(ex.map(_apply_model_to_file_enhanced, args),
                                 total=len(files), desc=f"Applying {method} model"):
                    if result is not None:
                        success_count += 1

        return success_count

    def calibrate(
        self,
        files: Sequence[str],
        calib_channels_dict: Optional[Dict[str, Sequence[float]]] = None,
    ) -> pd.DataFrame:
        """Calibrate all files using the selected calibration method."""
        if self.config.auto_tune:
            from ..adaptive import auto_tune_calib_config
            self.config = auto_tune_calib_config(
                files, self.config.reference_masses,
                base_config=self.config,
            )
            logger.info("auto_tune applied: tol_da=%s, breakpoints=%s",
                         self.config.autodetect_tol_da,
                         self.config.multisegment_breakpoints)

        os.makedirs(self.config.output_folder, exist_ok=True)
        original_ref_masses = np.asarray(self.config.reference_masses, dtype=float)
        ref_masses = _filter_reference_masses(
            original_ref_masses,
            self.config.exclude_reference_masses,
        )
        method = self.config.calibration_method
        self.last_reference_masses_initial = original_ref_masses.tolist()
        self.last_reference_masses_used = ref_masses.tolist()
        self.last_reference_masses_screened_out = []
        self.last_reference_mass_screening = pd.DataFrame()
        self.last_failed_or_excluded_files = pd.DataFrame()

        logger.info(f"Starting calibration with method={method} for {len(files)} files")

        if calib_channels_dict is None:
            logger.info("Autodetecting calibrant channels...")
            calib_channels_dict = self._autodetect_channels(files, ref_masses)
        elif len(ref_masses) != len(original_ref_masses):
            calib_channels_dict = _filter_calibration_values(
                calib_channels_dict,
                original_ref_masses,
                ref_masses,
            )

        results_map = self._fit_results_map(files, calib_channels_dict, ref_masses)

        if not results_map:
            raise RuntimeError(f"No {method} models could be fitted")

        logger.info(f"Successfully fitted {len(results_map)}/{len(files)} files")

        results_map = self._rescue_high_error_results_map(
            results_map,
            files,
            calib_channels_dict,
            ref_masses,
        )
        summary = self._build_summary(results_map)

        if self.config.auto_screen_reference_masses:
            screening_df = summarize_reference_mass_stability(
                summary,
                exclude_below_mz=self.config.screen_exclude_below_mz,
            )

            if self.config.auto_tune:
                from ..adaptive import estimate_screening_thresholds, estimate_outlier_threshold
                adaptive_screen = estimate_screening_thresholds(screening_df)
                if "screen_max_mean_abs_ppm" in adaptive_screen:
                    self.config = replace(self.config,
                                          screen_max_mean_abs_ppm=adaptive_screen["screen_max_mean_abs_ppm"])
                if "screen_min_valid_fraction" in adaptive_screen:
                    self.config = replace(self.config,
                                          screen_min_valid_fraction=adaptive_screen["screen_min_valid_fraction"])
                ppm_residuals = summary["ppm_error"].dropna().to_numpy(dtype=float)
                if ppm_residuals.size >= 6:
                    self.config = replace(self.config,
                                          outlier_threshold=estimate_outlier_threshold(ppm_residuals))
                logger.info("auto_tune screening: max_mean_abs_ppm=%.1f, min_valid_frac=%.2f, outlier_thr=%.2f",
                            self.config.screen_max_mean_abs_ppm,
                            self.config.screen_min_valid_fraction,
                            self.config.outlier_threshold)

            selected_ref_masses, screening_df = select_stable_reference_masses(
                ref_masses,
                screening_df,
                exclude_below_mz=self.config.screen_exclude_below_mz,
                max_mean_abs_ppm=self.config.screen_max_mean_abs_ppm,
                max_median_abs_ppm=self.config.screen_max_median_abs_ppm,
                min_valid_fraction=self.config.screen_min_valid_fraction,
                min_count=self.config.screen_min_count,
            )
            self.last_reference_mass_screening = screening_df

            selected_list = selected_ref_masses.tolist()
            screened_out = [
                float(mass) for mass in ref_masses
                if not np.isclose(selected_ref_masses, mass, atol=1e-6, rtol=0.0).any()
            ]
            self.last_reference_masses_screened_out = screened_out

            min_required = 2 if method == "linear_sqrt" else 3
            if len(selected_ref_masses) >= min_required and len(selected_ref_masses) < len(ref_masses):
                logger.info(
                    "Two-pass screening kept %d/%d reference masses; screened out=%s",
                    len(selected_ref_masses),
                    len(ref_masses),
                    screened_out,
                )
                calib_channels_dict = _filter_calibration_values(
                    calib_channels_dict,
                    ref_masses,
                    selected_ref_masses,
                )
                self.last_autodetect_methods = _filter_autodetect_methods(
                    self.last_autodetect_methods,
                    ref_masses,
                    selected_ref_masses,
                )
                ref_masses = selected_ref_masses
                self.last_reference_masses_used = selected_list
                results_map = self._fit_results_map(files, calib_channels_dict, ref_masses)
                if not results_map:
                    raise RuntimeError(f"No {method} models could be fitted after reference-mass screening")
                logger.info(
                    "Successfully refitted %d/%d files after reference-mass screening",
                    len(results_map),
                    len(files),
                )
                summary = self._build_summary(results_map)
            else:
                logger.info(
                    "Reference-mass screening did not change the working set "
                    "(selected=%d, initial=%d, minimum required=%d)",
                    len(selected_ref_masses),
                    len(ref_masses),
                    min_required,
                )
                self.last_reference_masses_used = ref_masses.tolist()
        else:
            self.last_reference_masses_used = ref_masses.tolist()

        fitted_files = set(results_map)
        missing_files = [
            os.path.basename(fp) for fp in files
            if os.path.basename(fp) not in fitted_files
        ]
        excluded_high_error_files: List[str] = []
        threshold_results_map = dict(results_map)

        if self.config.max_ppm_threshold:
            high_error_files = [
                fname for fname, rec in results_map.items()
                if not np.isfinite(rec["ppm"]) or rec["ppm"] > self.config.max_ppm_threshold
            ]
            if high_error_files:
                logger.warning(
                    f"{len(high_error_files)} files exceed PPM threshold "
                    f"({self.config.max_ppm_threshold:.1f})"
                )
                if self.config.fail_on_high_error:
                    excluded_high_error_files = list(high_error_files)
                    for fname in high_error_files:
                        results_map.pop(fname, None)
                    if not results_map:
                        raise RuntimeError(
                            f"No {method} models remain after excluding high-error files"
                        )
                    summary = self._build_summary(results_map)
                    logger.warning(
                        "Excluded %d files that remained above the %.1f ppm threshold",
                        len(high_error_files),
                        self.config.max_ppm_threshold,
                    )

        self.last_failed_or_excluded_files = self._build_issue_summary(
            missing_files=missing_files,
            results_map=threshold_results_map,
            excluded_high_error_files=excluded_high_error_files,
            threshold_ppm=self.config.max_ppm_threshold,
        )

        summary_path = os.path.join(
            self.config.output_folder,
            f"calibration_summary_{method}.tsv",
        )
        summary.to_csv(summary_path, sep="\t", index=False)
        if not self.last_failed_or_excluded_files.empty:
            issues_path = os.path.join(
                self.config.output_folder,
                f"calibration_issues_{method}.tsv",
            )
            self.last_failed_or_excluded_files.to_csv(issues_path, sep="\t", index=False)
        if self.config.auto_screen_reference_masses and not self.last_reference_mass_screening.empty:
            screening_path = os.path.join(
                self.config.output_folder,
                f"reference_mass_screening_{method}.tsv",
            )
            self.last_reference_mass_screening.to_csv(screening_path, sep="\t", index=False)

        # Apply calibration
        apply_files = [
            fp for fp in files
            if os.path.basename(fp) in results_map
        ]
        success_count = self._apply_results_map(apply_files, results_map)

        logger.info(f"Calibration complete: {success_count}/{len(files)} files processed")

        mean_ppm = summary["ppm_error"].mean()
        median_ppm = summary["ppm_error"].median()
        logger.info(f"Mean PPM error: {mean_ppm:.1f}, Median PPM error: {median_ppm:.1f}")

        return summary

    def _autodetect_channels(
        self,
        files: Sequence[str],
        ref_masses: npt.NDArray[np.float64],
    ) -> Dict[str, Sequence[float]]:
        """Autodetect calibrant channels with enhanced methods."""
        autodetected: Dict[str, Sequence[float]] = {}
        self.last_autodetect_methods = {}

        for fp in tqdm(files, desc="Autodetecting channels"):
            try:
                df = pd.read_csv(fp, sep="\t", header=0, comment="#")
            except Exception as e:
                logger.warning(f"{os.path.basename(fp)}: Failed to read - {e}")
                continue

            fname = os.path.basename(fp)

            if "Channel" not in df.columns:
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
            raise RuntimeError("Autodetection failed")

        logger.info(f"Autodetected channels for {len(autodetected)}/{len(files)} files")

        return autodetected
