import logging

import numpy as np
import pandas as pd
import tqdm
from .preprocessing import data_preprocessing

logger = logging.getLogger(__name__)

# Identify normalization target for ITC values across multiple files.

def normalization_target(
        files,
        mz_min=None,
        mz_max=None,
):
    """
    Normalize peak intensities or areas to a target value.

    Parameters
    ----------
    files : list of str
        List of file paths to process.
    mz_min, mz_max : float or None
        m/z window for data import (if supported).
    baseline_method : str
        Method for baseline correction.
    noise_method : str
        Noise filtering method.
    missing_value_method : str
        Method for handling missing values.

    Returns
    -------
    normalized_df : pd.DataFrame
        Normalized DataFrame.
    """
    tic_values = []
    for file_path in tqdm.tqdm(files):
        try:
            sample_name, group, mz_values, normalized_intensities = data_preprocessing(
                file_path=file_path,
                mz_min=mz_min,
                mz_max=mz_max,
                normalization_target=None,  # No normalization here
                verbose=False,
                return_all=False
            )
            tic = np.sum(normalized_intensities)
            tic_values.append((sample_name, group, (tic / 1e6).round(1)))
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    # Create a DataFrame with TIC values
    tic_df = pd.DataFrame(tic_values, columns=['SampleName', 'Group', 'TIC-Million'])
    logger.info(
        f"\n Mean TIC: {tic_df['TIC-Million'].mean().round(1)} Million"
        f"\n Median TIC: {tic_df['TIC-Million'].median().round(1)}  Million"
        f"\n Max TIC: {tic_df['TIC-Million'].max().round(1)}  Million"
        f"\n Min TIC: {tic_df['TIC-Million'].min().round(1)}  Million"
    )

    return tic_df