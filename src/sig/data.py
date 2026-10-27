"""Paths, labels and loaders shared by every script."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("SIG_DATA", ROOT / "data"))              # read-only inputs (Kaggle: /kaggle/input/...)
FEATURES = Path(os.environ.get("SIG_FEATURES", DATA / "features"))  # derived feature tables
CPUS = os.cpu_count() or 4
SEED = 42
LABELS = sorted([
    "eti_candidate", "frb", "noise", "pulsar", "radar_chirp",
    "rfi_intermittent", "rfi_persistent", "satellite_pass", "scintillating_star",
])
PRIORITY = [
    "eti_candidate", "frb", "pulsar", "scintillating_star", "satellite_pass",
    "radar_chirp", "rfi_intermittent", "rfi_persistent", "noise",
]
N_PANELS, N_ROWS, N_CHAN = 6, 16, 128
ON = np.array([True, False, True, False, True, False])


@dataclass
class Split:
    name: str
    x: np.ndarray        # (n, 6, 16, 128) uint8, 0 = dropped row
    meta: pd.DataFrame

    @property
    def gaps(self) -> np.ndarray:
        return self.meta["slew_gap_rows"].to_numpy()

    @property
    def y(self) -> np.ndarray:
        return self.meta["label"].map({c: i for i, c in enumerate(LABELS)}).to_numpy()


def load_split(name: str, mmap: bool = True) -> Split:
    x = np.load(DATA / f"{name}_cadences.npy", mmap_mode="r" if mmap else None)
    meta = pd.read_csv(DATA / f"{name}_meta.csv")
    return Split(name, x, meta)


def load_folds() -> np.ndarray:
    return pd.read_csv(ROOT / "folds.csv")["fold"].to_numpy()


def global_rows(gap: int) -> np.ndarray:
    """Global time step of every (panel, row): panel p, row r is at p * (16 + gap) + r."""
    return (np.arange(N_PANELS)[:, None] * (N_ROWS + gap) + np.arange(N_ROWS)[None, :])
