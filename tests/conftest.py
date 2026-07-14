"""Shared fixtures for the mioXpektron test suite."""

import numpy as np
import pytest


def make_spectrum(n=2000, mz_start=1.0, mz_step=0.05, peak_centers=(23.0, 39.0, 55.0),
                  peak_height=5000.0, peak_sigma=0.08, baseline=20.0, seed=0):
    """Return (mz, intensity) for a synthetic spectrum with Gaussian peaks."""
    rng = np.random.default_rng(seed)
    mz = mz_start + mz_step * np.arange(n)
    y = np.full(n, baseline, dtype=float)
    for c in peak_centers:
        y += peak_height * np.exp(-0.5 * ((mz - c) / peak_sigma) ** 2)
    y += rng.normal(0.0, 2.0, size=n)
    return mz, np.clip(y, 0.0, None)


def write_spectrum_txt(path, mz, intensity, channel=None, comment_lines=True):
    """Write a tab-separated spectrum file with Channel / m/z / Intensity columns."""
    lines = []
    if comment_lines:
        lines.append("# synthetic ToF-SIMS spectrum")
    if channel is None:
        channel = np.arange(len(mz))
    lines.append("Channel\tm/z\tIntensity")
    for ch, m, y in zip(channel, mz, intensity):
        lines.append(f"{int(ch)}\t{m:.6f}\t{y:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.fixture
def synthetic_spectrum():
    return make_spectrum()


@pytest.fixture
def spectrum_file(tmp_path, synthetic_spectrum):
    """A single synthetic spectrum written as a .txt file."""
    mz, intensity = synthetic_spectrum
    return write_spectrum_txt(tmp_path / "sample_CT-1a_1.txt", mz, intensity)


@pytest.fixture
def spectrum_files(tmp_path):
    """Several synthetic spectra with Cancer/Control naming."""
    paths = []
    for i, name in enumerate(
        ["breast_CT-1a_1.txt", "breast_CT-2b_1.txt", "breast_CC-1a_1.txt", "breast_CC-2b_1.txt"]
    ):
        mz, intensity = make_spectrum(seed=i, peak_height=4000.0 + 500.0 * i)
        paths.append(write_spectrum_txt(tmp_path / name, mz, intensity))
    return paths
