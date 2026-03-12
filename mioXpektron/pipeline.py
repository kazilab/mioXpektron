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
        reference_masses=(cfg.reference_masses or [
            1.00782503224, 22.9897692820, 38.9637064864, 58.065674,
            86.096974, 104.107539, 184.073871, 224.105171
        ]),
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


