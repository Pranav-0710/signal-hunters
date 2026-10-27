"""Hand-built cadence features from normalized cadences z (n, 6, 16, 128), see `sig.prep.normalize`.

Four searches, then the ON/OFF cadence logic on top of them:
- narrowband drift search per panel: S/N of the best straight line through the panel's mid row
  at every (drift, channel);
- global ETI track: one straight line through all six panels on the global time axis
  (`p * (16 + gap) + r`), scored separately on the ON and OFF panels;
- broadband dispersed sweeps per panel (pulsar, FRB, radar): sum over channels along a sweep
  that arrives later at lower frequencies;
- patch texture (scintillation): box-smoothed power over 4 rows x 8 channels.
"""

import numpy as np
import pandas as pd

from sig.data import N_CHAN, N_ROWS, ON

DRIFTS = np.concatenate([np.arange(-12, -6, 1.0), np.arange(-6, 6.01, 0.25), np.arange(7, 12.1, 1.0)])
PAD = 100
SMOOTH_VAR = 6 / 16                     # noise variance left after the [1, 2, 1] / 4 channel smoothing
TRACK_DRIFTS = np.arange(-1.5, 1.5001, 0.02)
STATIC = 0.06
SWEEP_SLOPES = np.array([2, 3, 4, 6, 8, 11, 15, 20, 26, 34], float)
SWEEP_T0 = np.arange(-34, 16)
GROUP = 4


def smooth_channels(z: np.ndarray) -> np.ndarray:
    p = np.pad(z, [(0, 0)] * (z.ndim - 1) + [(1, 1)], mode="edge")
    return 0.25 * p[..., :-2] + 0.5 * p[..., 1:-1] + 0.25 * p[..., 2:]


def drift_maps(z: np.ndarray, nvalid: np.ndarray) -> np.ndarray:
    """(n, 6, D, 128): S/N of a straight line with drift d (channels per row) crossing channel f at row 7.5."""
    zs = np.pad(smooth_channels(z), ((0, 0), (0, 0), (0, 0), (PAD, PAD)))
    r = np.arange(N_ROWS) - 7.5
    out = np.empty(z.shape[:2] + (len(DRIFTS), N_CHAN), np.float32)
    for i, d in enumerate(DRIFTS):
        sh = np.round(d * r).astype(int)
        acc = np.zeros(z.shape[:2] + (N_CHAN,), np.float32)
        for k in range(N_ROWS):
            acc += zs[:, :, k, PAD + sh[k]:PAD + sh[k] + N_CHAN]
        out[:, :, i] = acc
    return out / np.sqrt(np.maximum(nvalid, 1) * SMOOTH_VAR)[:, :, None, None]


def track_scores(s: np.ndarray, gaps: np.ndarray, chunk: int = 50) -> dict:
    parts = [_track_scores(s[a:a + chunk], gaps[a:a + chunk]) for a in range(0, len(s), chunk)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _track_scores(s: np.ndarray, gaps: np.ndarray) -> dict:
    """Score straight global-time tracks through the panel maps s (n, 6, D, 128).

    For a track with drift g crossing channel c at the middle of ON panel 3 (index 2), panel p is
    crossed at channel c + g * (p - 2) * (16 + gap). Each panel's S/N is read at its nearest local drift.
    """
    n = len(s)
    pooled = np.maximum(s, np.maximum(np.roll(s, 1, -1), np.roll(s, -1, -1)))
    di = np.abs(DRIFTS[None, :] - TRACK_DRIFTS[:, None]).argmin(1)                     # (G,)
    sub = pooled[:, :, di]                                                              # (n, 6, G, 128)
    dp = (np.arange(6) - 2)[None, :, None] * (N_ROWS + gaps)[:, None, None]             # (n, 6, 1)
    off = np.round(TRACK_DRIFTS[None, None, :] * dp).astype(np.int32)                       # (n, 6, G)
    pos = np.arange(N_CHAN)[None, None, None, :] + off[..., None]                       # (n, 6, G, 128)
    inside = (pos >= 0) & (pos < N_CHAN)
    vals = np.take_along_axis(sub, np.clip(pos, 0, N_CHAN - 1), axis=3)
    vals = np.where(inside, vals, np.float32(0))                                                 # out of band: no evidence
    on, offp = vals[:, ON], vals[:, ~ON]
    min_on = on.min(1)                                                                  # (n, G, 128)
    max_off = offp.max(1)
    clean = min_on - np.maximum(max_off, 0)
    static = np.abs(TRACK_DRIFTS) < STATIC
    feats = {}
    flat = min_on.reshape(n, -1)
    best = flat.argmax(1)
    g_best, c_best = np.unravel_index(best, min_on.shape[1:])
    feats["trk_min_on"] = flat.max(1)
    feats["trk_drift"] = TRACK_DRIFTS[g_best]
    feats["trk_abs_drift"] = np.abs(TRACK_DRIFTS[g_best])
    feats["trk_off_max_at_best"] = max_off[np.arange(n), g_best, c_best]
    feats["trk_off_mean_at_best"] = offp.mean(1)[np.arange(n), g_best, c_best]
    feats["trk_on_sum_at_best"] = on.sum(1)[np.arange(n), g_best, c_best] / np.sqrt(3)
    feats["trk_clean"] = clean.reshape(n, -1).max(1)
    feats["trk_clean_drifting"] = clean[:, ~static].reshape(n, -1).max(1)
    feats["trk_clean_static"] = clean[:, static].reshape(n, -1).max(1)
    feats["trk_min_on_drifting"] = min_on[:, ~static].reshape(n, -1).max(1)
    feats["trk_min_on_static"] = min_on[:, static].reshape(n, -1).max(1)
    feats["trk_sum_on"] = on.sum(1).reshape(n, -1).max(1) / np.sqrt(3)
    feats["trk_min_off"] = offp.min(1).reshape(n, -1).max(1)
    return feats


def sweep_maps(z: np.ndarray) -> np.ndarray:
    """(n, 6, shapes * slopes, t0): S/N of a broadband sweep arriving at row t0 at the top channel
    and `slope` rows later at the bottom, either linear or quadratic in channel."""
    n = len(z)
    g = z.reshape(n, 6, N_ROWS, N_CHAN // GROUP, GROUP).sum(-1) / np.sqrt(GROUP)          # (n, 6, 16, 32)
    ng = g.shape[-1]
    u = (ng - 1 - np.arange(ng)) / (ng - 1)                                                # 1 at the lowest channel
    out = []
    for shape in (u, u ** 2):
        for s in SWEEP_SLOPES:
            rows = np.round(SWEEP_T0[:, None] + s * shape[None, :]).astype(int)              # (T0, 32)
            ok = (rows >= 0) & (rows < N_ROWS)
            v = g[:, :, np.clip(rows, 0, N_ROWS - 1), np.arange(ng)[None, :]]              # (n, 6, T0, 32)
            out.append((v * ok).sum(-1) / np.sqrt(np.maximum(ok.sum(-1), 1)))
    return np.stack(out, 2)


def patch_texture(z: np.ndarray) -> np.ndarray:
    """(n, 6, 4, 16): mean S/N of 4-row x 8-channel boxes."""
    n = len(z)
    return z.reshape(n, 6, 4, 4, 16, 8).mean((3, 5)) * np.sqrt(32)


def _panel_block(name: str, v: np.ndarray, cols: dict) -> None:
    """Per-panel values v (n, 6) plus ON/OFF summaries."""
    for p in range(6):
        cols[f"{name}_p{p}"] = v[:, p]
    cols[f"{name}_on_mean"] = v[:, ON].mean(1)
    cols[f"{name}_off_mean"] = v[:, ~ON].mean(1)
    cols[f"{name}_on_min"] = v[:, ON].min(1)
    cols[f"{name}_off_max"] = v[:, ~ON].max(1)
    cols[f"{name}_on_off"] = cols[f"{name}_on_mean"] - cols[f"{name}_off_mean"]


def cadence_features(z: np.ndarray, valid: np.ndarray, gaps: np.ndarray) -> pd.DataFrame:
    n = len(z)
    cols: dict = {}
    nvalid = valid.sum(2)
    cols["n_dropped"] = (~valid).sum((1, 2))

    zv = z.reshape(n, 6, -1)
    _panel_block("std", zv.std(2), cols)
    _panel_block("frac_hi3", (zv > 3).mean(2), cols)
    _panel_block("frac_lo3", (zv < -3).mean(2), cols)
    _panel_block("p999", np.percentile(zv, 99.9, axis=2), cols)

    s = drift_maps(z, nvalid)                                                   # (n, 6, D, 128)
    static = np.abs(DRIFTS) < 0.13
    slow = (np.abs(DRIFTS) >= 0.25) & (np.abs(DRIFTS) <= 2)
    steep = np.abs(DRIFTS) >= 3
    best_d = s.max(3)                                                           # (n, 6, D)
    _panel_block("line_best", best_d.max(2), cols)
    _panel_block("line_static", best_d[:, :, static].max(2), cols)
    _panel_block("line_slow", best_d[:, :, slow].max(2), cols)
    _panel_block("line_steep", best_d[:, :, steep].max(2), cols)
    _panel_block("line_absdrift", np.abs(DRIFTS[best_d.argmax(2)]), cols)
    _panel_block("line_n_steep6", (s[:, :, steep].max(2) > 6).sum(2).astype(float), cols)
    _panel_block("line_n_static6", (s[:, :, static].max(2) > 6).sum(2).astype(float), cols)
    _panel_block("line_n_slow6", (s[:, :, slow].max(2) > 6).sum(2).astype(float), cols)
    cols["n_on_lines6"] = (best_d.max(2)[:, ON] > 6).sum(1)
    cols["n_off_lines6"] = (best_d.max(2)[:, ~ON] > 6).sum(1)
    cols["n_steep_panels8"] = (best_d[:, :, steep].max(2) > 8).sum(1)

    # Same channel in every panel: persistent RFI (allowing one faded panel via the second-lowest)
    st = s[:, :, static].max(2)                                                 # (n, 6, 128)
    st = np.maximum(st, np.maximum(np.roll(st, 1, -1), np.roll(st, -1, -1)))
    srt = np.sort(st, axis=1)
    cols["persist_min6"] = srt[:, 0].max(1)
    cols["persist_2nd6"] = srt[:, 1].max(1)
    cols["persist_on_min"] = st[:, ON].min(1).max(1)
    cols["persist_off_min"] = st[:, ~ON].min(1).max(1)
    cols["persist_n_ch"] = (srt[:, 1] > 5).sum(1)

    cols.update(track_scores(s, gaps))
    cols["trk_gap_vs_free"] = cols["trk_min_on"] - cols["line_best_on_min"]

    sw = sweep_maps(z)                                                          # (n, 6, K, T0)
    swb = sw.max(2)                                                             # (n, 6, T0)
    _panel_block("sweep_best", swb.max(2), cols)
    _panel_block("sweep_n4", (swb > 4).sum(2).astype(float), cols)
    k_best = sw.max(3).argmax(2)
    _panel_block("sweep_slope", np.tile(SWEEP_SLOPES, 2)[k_best], cols)
    _panel_block("sweep_curved", (k_best >= len(SWEEP_SLOPES)).astype(float), cols)
    # Repeating pulses: several separated peaks along t0 at the best sweep shape
    kb = np.take_along_axis(sw, k_best[..., None, None], 2)[:, :, 0]            # (n, 6, T0)
    peaks = (kb > 4) & (kb >= np.roll(kb, 1, -1)) & (kb >= np.roll(kb, -1, -1))
    _panel_block("sweep_npeaks", peaks.sum(2).astype(float), cols)

    tx = patch_texture(z).reshape(n, 6, -1)
    _panel_block("patch_std", tx.std(2), cols)
    _panel_block("patch_max", tx.max(2), cols)
    _panel_block("patch_n3", (tx > 3).sum(2).astype(float), cols)
    return pd.DataFrame(cols)
