from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import json
from datetime import datetime

import polars as pl
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

from .baseline_base import (
    baseline_correction,
    read_spectrum_table,
)

# ---------------------------------------------------------------------
# Batch corrector
# ---------------------------------------------------------------------

@dataclass
class BaselineBatchCorrector:
    in_dir: Union[str, Path]
    pattern: str = "*.csv"
    recursive: bool = False
    method: str = "airpls"
    method_kwargs: Dict = field(default_factory=dict)
    clip_negative: bool = True
    per_file_best: bool = False     # if True, expects a mapping file->method
    best_method_map: Optional[Dict[str, str]] = None  # file name -> method
    n_jobs: int = -1
    save_plots: bool = False

    def _timestamp_dir(self, root: Union[str, Path], prefix: str = "baseline_corrected_spectrum") -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out = Path(root) / f"{prefix}_{ts}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def run(self, out_root: Optional[Union[str, Path]] = None) -> Path:
        in_dir = Path(self.in_dir)
        glob_pat = f"**/{self.pattern}" if self.recursive else self.pattern
        files = sorted(in_dir.glob(glob_pat))
        if not files:
            raise FileNotFoundError(f"No files match {glob_pat} in {in_dir}")

        out_root = Path(out_root) if out_root else in_dir
        out_dir = self._timestamp_dir(out_root)

        def _proc(file: Path) -> Tuple[str, Optional[str]]:
            try:
                df = read_spectrum_table(file)
                y = df["intensity"].to_numpy()
                # Choose method
                m = (self.best_method_map.get(file.name) if (self.per_file_best and self.best_method_map) else self.method)
                y_corr, bl = baseline_correction(y, method=m, return_baseline=True,
                                                 clip_negative=self.clip_negative, **(self.method_kwargs or {}))
                out_df = df.clone()
                out_df = out_df.with_columns(
                    pl.Series("baseline", bl),
                    pl.Series("corrected_intensity", y_corr),
                )
                # save
                out_path = out_dir / (file.stem + "_baseline_corrected.csv")
                out_df.write_csv(out_path)
                # optional plot
                if self.save_plots:
                    x = out_df["mz"].to_numpy()
                    raw = out_df["intensity"].to_numpy()
                    baseline = out_df["baseline"].to_numpy()
                    corrected = out_df["corrected_intensity"].to_numpy()
                    plt.figure(figsize=(8.4, 4.2))
                    plt.plot(x, raw, lw=1, label="raw")
                    plt.plot(x, baseline, lw=1, linestyle=":", label="baseline")
                    plt.plot(x, corrected, lw=1, label="corrected")
                    plt.xlabel("m/z"); plt.ylabel("Intensity (a.u.)"); plt.title(file.name + f" | {m}")
                    plt.legend(frameon=False); plt.tight_layout()
                    for ext in (".pdf", ".png"):
                        plt.savefig(out_dir / (file.stem + f"_{m}_overlay{ext}"), bbox_inches="tight", dpi=300)
                    plt.close()

                return (file.name, None)
            except Exception as e:
                return (file.name, str(e))

        worker = delayed(_proc)
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(tqdm((worker(f) for f in files),
                                                                      total=len(files), desc="batch baseline", ncols=96))

        # manifest
        manifest = {
            "input_dir": str(in_dir.resolve()),
            "n_files": len(files),
            "method": self.method,
            "method_kwargs": self.method_kwargs,
            "clip_negative": self.clip_negative,
            "per_file_best": self.per_file_best,
            "errors": {fn: err for fn, err in results if err},
        }
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return out_dir
