# Data import function

import os, re
import polars as pl


def import_data(
        file_path: str,
        mz_min: float = None,
        mz_max: float = None
        ) -> tuple:
    """
    Import ToF-SIMS data from a text file.

    Parameters
    ----------
    file_path : str
        Path to the ToF-SIMS data file
    mz_min : float, optional
        Minimum m/z value to be imported (inclusive)
    mz_max : float, optional
        Maximum m/z value to be imported (inclusive)

    Returns
    -------
    mz : np.ndarray
        Mass-to-charge ratio values
    intensity : np.ndarray
        Intensity values
    sample_name : str
        Sample name extracted from file name
    group : str
        'cancer' or 'control' based on file path
    """

    # Read file, skip lines starting with '#'
    df = pl.read_csv(file_path, separator="\t", comment_prefix="#")

    # Check required columns
    if not {'m/z', 'Intensity'}.issubset(df.columns):
        raise ValueError(f"File {file_path} missing required columns.")

    # Apply m/z filtering (inclusive)
    if mz_min is not None:
        df = df.filter(pl.col('m/z') >= mz_min)
    if mz_max is not None:
        df = df.filter(pl.col('m/z') <= mz_max)

    if df.height == 0:
        raise ValueError(f"No data in {file_path} after m/z filtering.")

    mz = df['m/z'].to_numpy()
    intensity = df['Intensity'].to_numpy()
    sample_name = os.path.basename(file_path).replace('.txt', '')
    if re.search(r'_CC(?=_|\b)', sample_name, flags=re.IGNORECASE):
        group = 'Cancer'
    elif re.search(r'_CT(?=_|\b)', sample_name, flags=re.IGNORECASE):
        group = 'Control'
    else:
        group = 'Unknown'
    return mz, intensity, sample_name, group
