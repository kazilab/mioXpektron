import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .recalibrate.auto_calibrator import AutoCalibrator, AutoCalibConfig
from .denoise.denoise_main import noise_filtering
from .baseline.baseline_base import baseline_correction
from .utils.file_management import import_data
from .normalization.normalization import tic_normalization
from .utils.main import batch_processing

DEFAULT_REFERENCE_MASSES: List[float] = [
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


@dataclass
class PipelineConfig:
    """High-level pipeline configuration for batch ToF‑SIMS processing."""
    # Recalibration
    use_recalibration: bool = True
    reference_masses: Optional[List[float]] = None
    output_folder_calibrated: str = "calibrated_spectra"

    # Denoising (simple single-method apply; selection can be integrated later)
    denoise_method: str = "wavelet"  # {"wavelet","gaussian","median","savitzky_golay","none"}
    denoise_params: Optional[Dict] = None

    # Baseline correction
    baseline_method: str = "airpls"
    baseline_params: Optional[Dict] = None
    clip_negative_after_baseline: bool = True

    # Normalization (TIC via preprocessing in import path; can be extended)
    normalization_target: float = 1e6

    # Peak/Alignment
    mz_min: Optional[float] = None
    mz_max: Optional[float] = None
    mz_tolerance: float = 0.2
    mz_rounding_precision: int = 1

    # Parallelism
    max_workers: Optional[int] = None

    # Adaptive parameterization (opt-in)
    auto_tune: bool = False


def _maybe_recalibrate(
    files: Sequence[str],
    calib_channels_dict: Optional[Dict[str, Sequence[float]]],
    cfg: PipelineConfig,
) -> List[str]:
    """Optionally run recalibration and return paths to calibrated spectra.

    If ``use_recalibration`` is False or no calibration data is provided,
    returns the original ``files`` list.
    """
    if not cfg.use_recalibration or not calib_channels_dict:
        return list(files)

    cal_cfg = AutoCalibConfig(
        reference_masses=(cfg.reference_masses or DEFAULT_REFERENCE_MASSES),
        output_folder=cfg.output_folder_calibrated,
        max_workers=cfg.max_workers,
    )
    calibrator = AutoCalibrator(config=cal_cfg)
    _ = calibrator.calibrate(list(files), calib_channels_dict)

    # Return paths to newly written calibrated spectra
    out_files: List[str] = []
    for fp in files:
        base = os.path.basename(fp)
        out_files.append(os.path.join(cfg.output_folder_calibrated, base.replace(".txt", "_calibrated.txt")))
    return out_files


def _load_apply_denoise_baseline_normalize(
    file_path: str,
    cfg: PipelineConfig,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Load a spectrum, apply denoise, baseline correction, normalization.

    Returns (sample_name, mz_values, intensities_processed).
    """
    mz, intensity, sample_name, _group = import_data(
        file_path=file_path,
        mz_min=cfg.mz_min,
        mz_max=cfg.mz_max,
    )

    # Denoise
    y = intensity.astype(float)
    if cfg.denoise_method and cfg.denoise_method != "none":
        y = noise_filtering(y, method=cfg.denoise_method, **(cfg.denoise_params or {}))

    # Baseline correction
    y = baseline_correction(
        y,
        method=cfg.baseline_method,
        clip_negative=cfg.clip_negative_after_baseline,
        **(cfg.baseline_params or {}),
    )

    # Normalization (TIC)
    y = tic_normalization(y, target_tic=cfg.normalization_target)

    return sample_name, mz, y


def run_pipeline(
    files: Sequence[str],
    *,
    calib_channels_dict: Optional[Dict[str, Sequence[float]]] = None,
    config: Optional[PipelineConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the end‑to‑end ToF‑SIMS batch pipeline and return aligned matrices.

    Steps
    -----
    1) Optional recalibration (Channel→m/z)
    2) Denoising
    3) Baseline correction
    4) TIC normalization
    5) Peak detection and alignment → unified m/z × samples tables

    Returns
    -------
    (intensity_df, area_df) aligned by m/z across samples.
    """
    cfg = config or PipelineConfig()

    if cfg.auto_tune:
        from .adaptive import estimate_mz_tolerance, estimate_normalization_target
        cfg.mz_tolerance = estimate_mz_tolerance(list(files))
        cfg.normalization_target = estimate_normalization_target(
            list(files), mz_min=cfg.mz_min, mz_max=cfg.mz_max,
        )

    # 1) Optional recalibration
    working_files = _maybe_recalibrate(files, calib_channels_dict, cfg)

    # 2–5) Peak detection + alignment via batch utility.
    #    batch_processing handles its own preprocessing (denoise, baseline, normalization)
    #    internally via data_preprocessing → detect → align.
    _peaks_df, intensity_df, area_df = batch_processing(
        working_files,
        max_workers=cfg.max_workers,
        mz_min=cfg.mz_min,
        mz_max=cfg.mz_max,
        normalization_target=cfg.normalization_target,
        method=None,  # None → detect_peaks_with_area (local-max); "cwt" → CWT detector
        mz_tolerance=cfg.mz_tolerance,
        mz_rounding_precision=cfg.mz_rounding_precision,
    )

    return intensity_df, area_df


