"""Telescope-neutral normalization of a batch of cadences.

Steps, all per cadence (n, 6, 16, 128) in log2-power units (byte / 24):
1. Dropped rows (all zeros) become missing.
2. Subtract each row's median over channels: removes panel gain changes and broadband glitch rows.
3. Subtract a smooth bandpass: the per-channel median over all 96 rows, running-median smoothed
   across channels. Narrow lines survive (a persistent RFI line is too thin to bend the smooth
   curve); ripple, roll-off and tilt do not.
4. The GBT centre-channel spike is replaced by its neighbours on every cadence, so neither
   telescope has it.
5. Divide by the robust noise level of the cadence.
"""

import numpy as np
from scipy.ndimage import median_filter

SCALE = 24.0
SMOOTH = 13
SPIKE = 64


PKS_GAMMA_SHAPE = 7.7     # extra multiplicative noise that widens GBT byte MAD 13.3 to the PKS 18.5
PKS_EXTRA_DROP = 0.0023   # PKS drops 0.38% of rows, GBT 0.15%


def degrade(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make GBT cadences look like PKS noise: multiply linear power by Gamma(k)/k noise, drop rows.

    Multiplying power by an independent gamma variable is what averaging fewer spectra does, so
    the noise gets wider and gains the negative log-domain skew seen in PKS, and signals fade with it.
    """
    g = rng.gamma(PKS_GAMMA_SHAPE, 1 / PKS_GAMMA_SHAPE, size=x.shape).astype(np.float32)
    v = x.astype(np.float32) + SCALE * np.log2(g)
    out = np.clip(np.round(v), 1, 255).astype(np.uint8)
    drop = rng.random(x.shape[:3]) < PKS_EXTRA_DROP
    out[(x == 0) | drop[..., None]] = 0
    return out


def normalize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (z, valid): z float32 (n, 6, 16, 128) with 0 at dropped rows; valid (n, 6, 16) bool."""
    f = x.astype(np.float32) / SCALE
    valid = x.max(axis=3) > 0
    f[..., SPIKE] = 0.5 * (f[..., SPIKE - 1] + f[..., SPIKE + 1])

    f -= np.median(f, axis=3, keepdims=True)
    f[~valid] = 0.0

    n = len(f)
    flat = f.reshape(n, -1, f.shape[-1])
    w = valid.reshape(n, -1)
    prof = np.median(flat, axis=1)
    gappy = np.where(~w.all(axis=1))[0]
    if len(gappy):
        prof[gappy] = np.nanmedian(np.where(w[gappy, :, None], flat[gappy], np.nan), axis=1)
    bp = median_filter(prof, size=(1, SMOOTH), mode="nearest")
    f -= bp[:, None, None, :]
    f[~valid] = 0.0

    sig = 1.4826 * np.median(np.abs(f.reshape(n, -1)), axis=1) + 1e-6
    z = f / sig[:, None, None, None]
    z[~valid] = 0.0
    return z.astype(np.float32), valid
