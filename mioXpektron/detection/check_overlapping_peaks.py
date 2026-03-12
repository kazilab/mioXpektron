import glob
import os
import polars as pl
import numpy as np

from ..normalization.normalization import tic_normalization
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm   # optional for the 2-D histogram

def load_window(file_path, mz_min, mz_max, norm_tic=False):
    """Read one spectrum and return m/z, intensity in the requested window."""
    try:
        df = pl.read_csv(
            file_path,
            separator="\t",
            comment_prefix="#"
        ).filter(
            (pl.col("m/z") >= mz_min) & (pl.col("m/z") <= mz_max)
        )
    except TypeError:
        # Polars version that uses comment_char instead of comment_prefix
        df = pl.read_csv(
            file_path,
            separator="\t",
            comment_char="#"
        ).filter(
            (pl.col("m/z") >= mz_min) & (pl.col("m/z") <= mz_max)
        )

    mz  = df["m/z"].to_numpy()
    inten = df["Intensity"].to_numpy()
    if norm_tic:
        inten = tic_normalization(inten, target_tic=1e6)
    return mz, inten


def check_overlapping_peaks(data_dir, file_name, mz_min, mz_max, norm_tic=False, alpha=0.2 ):
    # 1) gather files
    files = sorted(glob.glob(os.path.join(data_dir, file_name)))
    if len(files) == 0:
        raise RuntimeError("No *.txt files found in DATA_DIR")

    # 2) read everything
    all_curves = []
    for fp in files:
        mz, inten = load_window(fp, mz_min, mz_max, norm_tic=norm_tic)
        if len(mz) == 0:  # skip empty window
            continue
        all_curves.append((mz, inten))

    # 3) common m/z grid (needed only for the 2-D histogram or if curves differ)
    #    here we oversample to 0.001 m/z bins for smoothness
    mz_common = np.arange(mz_min, mz_max, 0.001)
    interp_curves = []
    for mz, inten in all_curves:
        interp_inten = np.interp(mz_common, mz, inten, left=0, right=0)
        interp_curves.append(interp_inten)

    # 4) line-overlay plot
    plt.figure(figsize=(10, 6))
    for mz, inten in all_curves:
        plt.plot(mz, inten, linewidth=0.8, alpha=alpha)
    plt.xlim(mz_min, mz_max)
    plt.xlabel("m/z")
    plt.ylabel("Normalised intensity" if norm_tic else "Intensity")
    plt.title(f"All {len(files)} spectra overlaid ({mz_min}–{mz_max} m/z)")
    plt.tight_layout()
    plt.show()

    # 5) optional density heat-map (2-D histogram)
    #    This collapses the 96 lines into a single image where colour = frequency
    counts = np.vstack(interp_curves)  # shape (n_spectra, n_bins)
    density = counts.sum(axis=0)  # simple sum; could use log10

    plt.figure(figsize=(10, 3))
    plt.bar(mz_common, density, width=0.001, color="k")
    plt.xlim(mz_min, mz_max)
    plt.xlabel("m/z")
    plt.ylabel("Cumulative intensity")
    plt.title("Cumulative intensity (sum of all spectra)")
    plt.tight_layout()
    plt.show()

    # Alternatively, for a true heat map:
    plt.figure(figsize=(10, 6))
    plt.imshow(
        counts,
        aspect="auto",
        extent=[mz_min, mz_max, 0, len(all_curves)],
        origin="lower",
        norm=LogNorm()  # highlights both weak and strong overlaps
    )
    plt.colorbar(label="Normalised intensity")
    plt.xlabel("m/z")
    plt.ylabel("Spectrum index")
    plt.title(f"Intensity heat-map ({mz_min}–{mz_max} m/z)")
    plt.tight_layout()
    plt.show()