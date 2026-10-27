"""Compute hand-built features for train, PKS-degraded train (seed 42) and test.

Writes data/features/{train,train_deg,test}.parquet, one row per cadence in file order.
    python scripts/build_features.py
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sig.data import CPUS, FEATURES, SEED, load_split  # noqa: E402
from sig.features import cadence_features  # noqa: E402
from sig.prep import degrade, normalize  # noqa: E402

CHUNK = 250
OUT = FEATURES


def _work(args: tuple) -> pd.DataFrame:
    name, start, stop, deg = args
    sp = load_split(name)
    x = np.asarray(sp.x[start:stop])
    if deg:
        x = degrade(x, np.random.default_rng([SEED, start]))
    z, valid = normalize(x)
    return cadence_features(z, valid, sp.gaps[start:stop])


def build(name: str, deg: bool, workers: int, limit: int | None) -> pd.DataFrame:
    n = len(load_split(name).meta) if limit is None else limit
    jobs = [(name, a, min(a + CHUNK, n), deg) for a in range(0, n, CHUNK)]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_work, jobs))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=CPUS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, deg, tag in [("train", False, "train"), ("train", True, "train_deg"), ("test", False, "test")]:
        t = time.time()
        f = build(name, deg, args.workers, args.limit)
        f.to_parquet(OUT / f"{tag}.parquet")
        print(f"{tag}: {f.shape} in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
