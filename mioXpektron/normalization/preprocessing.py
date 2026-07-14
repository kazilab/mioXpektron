# Data import function

import logging
import os
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator

logger = logging.getLogger(__name__)

from ..utils.file_management import import_data
from .normalization import tic_normalization

_RESAMPLE_METHODS = ("linear", "pchip", "akima", "makima", "cubic")


def _build_akima_interpolator(mz_values, intensity_values, method):
    """Build an Akima-family interpolator across SciPy versions."""
    if method == "makima":
        try:
            return Akima1DInterpolator(
                mz_values,
                intensity_values,
                method="makima",
                extrapolate=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "resample method 'makima' requires scipy>=1.13.0."
            ) from exc

    try:
        return Akima1DInterpolator(
            mz_values,
            intensity_values,
            extrapolate=False,
        )
    except TypeError:
        return Akima1DInterpolator(mz_values, intensity_values)


def resample_spectrum(mz_values, intensity_values, target_mz, method="linear"):
    """Resample a spectrum onto a target m/z grid.

    The input axis is sorted, duplicate m/z positions are collapsed to their
    first occurrence, and values outside the native m/z range are filled with
    zero. Supported interpolation methods are ``linear``, ``pchip``,
    ``akima``, ``makima``, and ``cubic``.
    """
    mz_values = np.asarray(mz_values, dtype=float).reshape(-1)
    intensity_values = np.asarray(intensity_values, dtype=float).reshape(-1)
    target_mz = np.asarray(target_mz, dtype=float)

    if mz_values.size != intensity_values.size:
        raise ValueError(
            "mz_values and intensity_values must have the same length."
        )
    if mz_values.size == 0:
        return np.zeros_like(target_mz, dtype=float)

    order = np.argsort(mz_values)
    mz_values = mz_values[order]
    intensity_values = intensity_values[order]

    mz_values, unique_idx = np.unique(mz_values, return_index=True)
    intensity_values = intensity_values[unique_idx]

    if method == "linear" or mz_values.size == 1:
        resampled = np.interp(
            target_mz,
            mz_values,
            intensity_values,
            left=0.0,
            right=0.0,
        )
        return np.asarray(resampled, dtype=float)

    if method == "pchip":
        interpolator = PchipInterpolator(
            mz_values,
            intensity_values,
            extrapolate=False,
        )
    elif method in {"akima", "makima"}:
        interpolator = _build_akima_interpolator(
            mz_values,
            intensity_values,
            method,
        )
    elif method == "cubic":
        interpolator = CubicSpline(
            mz_values,
            intensity_values,
            extrapolate=False,
        )
    else:
        valid_methods = ", ".join(_RESAMPLE_METHODS)
        raise ValueError(
            f"Unknown resample_method={method!r}. Use one of: {valid_methods}."
        )

    resampled = interpolator(target_mz)
    resampled = np.nan_to_num(resampled, nan=0.0, posinf=0.0, neginf=0.0)
    outside = (target_mz < mz_values[0]) | (target_mz > mz_values[-1])
    resampled = np.where(outside, 0.0, resampled)
    resampled = np.clip(resampled, 0.0, None)

    return np.asarray(resampled, dtype=float)


def data_preprocessing(
        file_path,
        mz_min=None,
        mz_max=None,
        normalization_target=1e6,
        verbose=True,
        return_all=False
    ):
    """
    Import and preprocess ToF-SIMS data from a text file.

    Parameters:
    -----------
    file_path : str
        Path to the ToF-SIMS data file
    mz_min, mz_max : float, optional
        m/z range to import
    normalization_target : float or None
        Target TIC for normalization, or None to skip
    verbose : bool
        Print progress if True
    return_all : bool
        If True, return all intermediate arrays

    Returns:
    --------
    mz_values : numpy.ndarray
    normalized_intensities : numpy.ndarray
    sample_name : str
    group : str
    (optionally: intermediate arrays)
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    mz_values, intensity, sample_name, group = import_data(file_path, mz_min, mz_max)
    if verbose: logger.info(f"Imported data: {sample_name} {mz_values.shape} {intensity.shape}")

    # TIC normalization
    if normalization_target:
        normalized_intensities = tic_normalization(intensity, target_tic=normalization_target)
        if verbose: logger.info(f"TIC normalized.")
    else:
        normalized_intensities = intensity

    if return_all:
        return (sample_name, group, mz_values,  
                intensity, normalized_intensities)
    else:
        return sample_name, group, mz_values, normalized_intensities


# Batch preprocessing helper
def batch_tic_norm(input_pattern: str,
                     output_dir: str = "normalized_spectra",
                     mz_min: Optional[float] = None,
                     mz_max: Optional[float] = None,
                     normalization_target: Optional[float] = 1e6,
                     verbose: bool = False) -> List[str]:
    """
    Batch‑import and preprocess multiple ToF‑SIMS spectra, then save the
    (m/z, normalized_intensity) arrays for each file as a tab‑separated text
    file in *output_dir*.

    Parameters
    ----------
    input_pattern : str
        Glob pattern (e.g. 'spectra/*.txt') that expands to the input files.
    output_dir : str
        Folder where '<original‑name>_normalized.txt' will be written;
        created if it does not already exist.
    mz_min, mz_max, normalization_target, verbose
        Passed through to :pyfunc:`data_preprocessing`.
    Returns
    -------
    List[str]
        Paths of the files written, in processing order.
    """
    import glob
    import numpy as np

    files = sorted(glob.glob(input_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{input_pattern}'")

    os.makedirs(output_dir, exist_ok=True)

    written: List[str] = []
    for fp in files:
        sample_name, group, mz_vals, norm_intens = data_preprocessing(
            fp,
            mz_min=mz_min,
            mz_max=mz_max,
            normalization_target=normalization_target,
            verbose=verbose,
            return_all=False,
        )

        base = os.path.splitext(os.path.basename(fp))[0]
        out_path = os.path.join(output_dir, f"{base}_normalized.txt")

        np.savetxt(
            out_path,
            np.column_stack((mz_vals, norm_intens)),
            fmt="%.6f\t%.6e",
            header="m/z\tIntensity",
            comments=""
        )
        if verbose:
            logger.info(f"Wrote {out_path}")
        written.append(out_path)

    return written


class BatchTicNorm:
    """
    Batch TIC normalization for multiple spectra files using Polars and concurrent.futures.

    Supports both CSV and TXT file formats:
    - CSV: Uses 'corrected_intensity' if available, otherwise 'intensity'
    - TXT: Tab-separated m/z and intensity values

    Output files contain: channel, mz, intensity (normalized)
    """

    def __init__(
        self,
        input_pattern: str,
        output_dir: str = "normalized_spectra",
        normalization_target: float = 1e6,
        n_workers: int = -1,
        verbose: bool = True
    ):
        """
        Initialize BatchTicNorm processor.

        Parameters
        ----------
        input_pattern : str
            Glob pattern for input files (e.g., 'data/*.csv' or 'data/*.txt')
        output_dir : str
            Directory to save normalized files
        normalization_target : float
            Target TIC value for normalization (default: 1e6)
        n_workers : int
            Number of parallel workers (default: 16)
        verbose : bool
            Print progress information
        """
        self.input_pattern = input_pattern
        self.output_dir = Path(output_dir)
        self.normalization_target = normalization_target
        self.n_workers = cpu_count() if n_workers <= 0 else min(n_workers, cpu_count())
        self.verbose = verbose

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find input files
        self.input_files = sorted(glob.glob(input_pattern))
        if not self.input_files:
            raise FileNotFoundError(f"No files matched pattern: {input_pattern}")

        if self.verbose:
            logger.info(f"Found {len(self.input_files)} files to process")
            logger.info(f"Using {self.n_workers} workers")

    def _read_file(self, file_path: str) -> pl.DataFrame:
        """
        Read a single file (CSV or TXT) and return a Polars DataFrame.

        Parameters
        ----------
        file_path : str
            Path to the input file

        Returns
        -------
        pl.DataFrame
            DataFrame with columns: channel, mz, intensity
        """
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.csv':
            # Read CSV file
            df = pl.read_csv(file_path)

            # Select intensity column (prefer corrected_intensity if available)
            if 'corrected_intensity' in df.columns:
                intensity_col = 'corrected_intensity'
            elif 'intensity' in df.columns:
                intensity_col = 'intensity'
            else:
                raise ValueError(f"No intensity column found in {file_path}")

            # Check if channel column exists
            if 'channel' not in df.columns:
                # Create channel column if missing
                df = df.with_row_index(name='channel')

            # Select and rename columns
            result = df.select([
                pl.col('channel'),
                pl.col('mz'),
                pl.col(intensity_col).alias('intensity')
            ])

        elif file_ext in ['.txt', '.tsv']:
            # Read TXT/TSV file (assumes tab-separated m/z and intensity)
            df = pl.read_csv(
                file_path,
                separator='\t',
                has_header=True,
                skip_rows_after_header=0
            )

            # If no header or only 2 columns, assume m/z and intensity
            if df.shape[1] == 2:
                df.columns = ['mz', 'intensity']

            # Add channel column
            result = df.with_row_index(name='channel').select([
                pl.col('channel'),
                pl.col('mz'),
                pl.col('intensity')
            ])

        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        return result

    def _normalize_single_file(self, file_path: str) -> Tuple[str, bool]:
        """
        Process and normalize a single file.

        Parameters
        ----------
        file_path : str
            Path to the input file

        Returns
        -------
        Tuple[str, bool]
            (output_path, success)
        """
        try:
            # Read file
            df = self._read_file(file_path)

            # Extract intensity as numpy array for normalization
            intensities = df['intensity'].to_numpy()

            # Perform TIC normalization
            normalized_intensities = tic_normalization(
                intensities,
                target_tic=self.normalization_target
            )

            # Create output DataFrame
            output_df = df.with_columns(
                pl.Series('intensity', normalized_intensities)
            )

            # Generate output path
            base_name = Path(file_path).stem
            output_path = self.output_dir / f"{base_name}_normalized.csv"

            # Write to CSV
            output_df.write_csv(output_path)

            if self.verbose:
                tic_before = np.sum(intensities)
                tic_after = np.sum(normalized_intensities)
                logger.info(f"✓ {Path(file_path).name}: TIC {tic_before:.2e} → {tic_after:.2e}")

            return str(output_path), True

        except Exception as e:
            if self.verbose:
                logger.error(f"✗ Error processing {Path(file_path).name}: {str(e)}")
            return "", False

    def process(self) -> List[str]:
        """
        Process all files using concurrent.futures.

        Returns
        -------
        List[str]
            List of output file paths that were successfully created
        """
        if self.verbose:
            logger.info(f"Processing {len(self.input_files)} files...")
            logger.info(f"Normalization target: {self.normalization_target:.2e}")
            logger.info(f"Output directory: {self.output_dir}")

        # Process files in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(self._normalize_single_file, self.input_files))

        # Collect successful outputs
        output_files = [path for path, success in results if success]

        if self.verbose:
            logger.info(f"Completed: {len(output_files)}/{len(self.input_files)} files normalized")
            logger.info(f"Output location: {self.output_dir.absolute()}")

        return output_files

    def get_tic_statistics(self) -> pl.DataFrame:
        """
        Calculate TIC statistics for all input files before normalization.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns: filename, tic_original, tic_million
        """
        stats = []

        for file_path in self.input_files:
            try:
                df = self._read_file(file_path)
                tic = df['intensity'].sum()
                stats.append({
                    'filename': Path(file_path).name,
                    'tic_original': tic,
                    'tic_million': tic / 1e6
                })
            except Exception as e:
                if self.verbose:
                    logger.error(f"Error reading {file_path}: {e}")

        stats_df = pl.DataFrame(stats)

        if self.verbose and len(stats) > 0:
            logger.info("TIC Statistics (before normalization):")
            logger.info(f"  Mean TIC:   {stats_df['tic_million'].mean():.2f} Million")
            logger.info(f"  Median TIC: {stats_df['tic_million'].median():.2f} Million")
            logger.info(f"  Min TIC:    {stats_df['tic_million'].min():.2f} Million")
            logger.info(f"  Max TIC:    {stats_df['tic_million'].max():.2f} Million")

        return stats_df
