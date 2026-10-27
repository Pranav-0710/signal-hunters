"""Blend model probabilities and write submissions/<out>.csv.

    python scripts/make_submission.py --models lgbm --out 01_lgbm
    python scripts/make_submission.py --models lgbm cnn --weights 0.5 0.5 --out 02_blend
    python scripts/make_submission.py --models lgbm cnn --em 0.5 --out 03_blend_em
    python scripts/make_submission.py --models stack4 --bias --out 10_stack4_bias

Blend: weighted mean of log-probabilities, renormalized. Every model is trained with balanced
class weights, so its implied class prior is uniform.

--em ALPHA estimates the test class mix from the unlabeled test predictions with the
Saerens-Latinne-Decaestecker EM (no labels involved), then multiplies the posteriors by
(test prior / uniform) ** ALPHA before the argmax. ALPHA 0 is off, 1 is the full Bayes shift.

--bias adds one log-probability offset per class, chosen by coordinate ascent to maximize OOF
macro-F1 (macro-F1 rewards trading a little accuracy on easy classes for recall on hard ones).
The gain is first measured honestly: offsets tuned on 4 folds, scored on the 5th. Test uses
offsets tuned on all OOF rows.

--shift ALPHA is meant for use with --bias: the OOF-tuned offsets carry the train class mix, so
this adds ALPHA * log(EM test prior / train class share) to move them toward the test mix instead.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sig.data import DATA, LABELS, ROOT, load_folds, load_split  # noqa: E402

BIAS_GRID = np.linspace(-2, 2, 41)


def blend(models: list[str], weights: list[float], part: str) -> tuple[pd.Series, np.ndarray]:
    logp, ids = 0.0, None
    for m, w in zip(models, weights):
        f = pd.read_csv(ROOT / f"oof/{m}_{part}.csv")
        ids = f["id"]
        logp = logp + w * np.log(np.clip(f[LABELS].to_numpy(), 1e-9, 1))
    p = np.exp(logp - logp.max(1, keepdims=True))
    return ids, p / p.sum(1, keepdims=True)


def em_prior(p: np.ndarray, iters: int = 200) -> np.ndarray:
    k = p.shape[1]
    prior = np.full(k, 1 / k)
    for _ in range(iters):
        q = p * prior * k
        q /= q.sum(1, keepdims=True)
        prior = q.mean(0)
    return prior


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    return float(np.mean(2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)))


def tune_bias(logp: np.ndarray, y: np.ndarray, rounds: int = 3) -> np.ndarray:
    k = logp.shape[1]
    b = np.zeros(k)
    best = macro_f1(y, logp.argmax(1), k)
    for _ in range(rounds):
        for c in range(k):
            for v in BIAS_GRID:
                trial = b.copy()
                trial[c] = v
                s = macro_f1(y, (logp + trial).argmax(1), k)
                if s > best + 1e-6:
                    best, b = s, trial
    return b - b.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--em", type=float, default=0.0)
    ap.add_argument("--bias", action="store_true", help="tune per-class log offsets on OOF macro-F1")
    ap.add_argument("--shift", type=float, default=0.0, help="move offsets from the train mix toward the EM test mix")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    weights = args.weights or [1 / len(args.models)] * len(args.models)

    y = load_split("train").y
    _, oof = blend(args.models, weights, "oof")
    ok = ~np.isnan(oof).any(1)
    print(f"OOF macro-F1 (degraded train, {ok.sum()} rows): "
          f"{f1_score(y[ok], oof[ok].argmax(1), average='macro'):.4f}")

    bias = np.zeros(len(LABELS))
    if args.bias:
        logp, folds = np.log(np.clip(oof, 1e-9, 1)), load_folds()
        pred = np.zeros(len(y), dtype=int)
        for f in range(5):
            fit, val = (folds != f) & ok, (folds == f) & ok
            pred[val] = (logp[val] + tune_bias(logp[fit], y[fit])).argmax(1)
        print(f"OOF macro-F1 with bias (tuned on other folds): {f1_score(y[ok], pred[ok], average='macro'):.4f}")
        bias = tune_bias(logp[ok], y[ok])
        print("bias:", dict(zip(LABELS, bias.round(2))))
        print("per-class F1 with bias:", dict(zip(LABELS, f1_score(y[ok], pred[ok], average=None).round(3))))

    ids, p = blend(args.models, weights, "test")
    prior = em_prior(p)
    print("EM test prior:", dict(zip(LABELS, prior.round(3))))
    if args.em:
        p = p * (prior * len(LABELS)) ** args.em
    if args.shift:
        bias = bias + args.shift * np.log(prior / (np.bincount(y, minlength=len(LABELS)) / len(y)))
        print("offsets after shift:", dict(zip(LABELS, bias.round(2))))
    pred = np.array(LABELS)[(np.log(np.clip(p, 1e-9, 1)) + bias).argmax(1)]

    sub = pd.DataFrame({"id": ids, "label": pred})
    sample = pd.read_csv(DATA / "sample_submission.csv")
    assert len(sub) == len(sample) and set(sub["id"]) == set(sample["id"]) and sub["id"].is_unique
    sub = sample[["id"]].merge(sub, on="id", how="left")
    assert sub["label"].isin(LABELS).all()
    (ROOT / "submissions").mkdir(exist_ok=True)
    sub.to_csv(ROOT / f"submissions/{args.out}.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print("predicted mix:", sub["label"].value_counts().to_dict())
    print(f"wrote submissions/{args.out}.csv at commit {commit}")


if __name__ == "__main__":
    main()
