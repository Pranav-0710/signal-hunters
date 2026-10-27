"""Blend model probabilities and write submissions/<out>.csv.

    python scripts/make_submission.py --models lgbm --out 01_lgbm
    python scripts/make_submission.py --models lgbm cnn --weights 0.5 0.5 --out 02_blend
    python scripts/make_submission.py --models lgbm cnn --em 0.5 --out 03_blend_em

Blend: weighted mean of log-probabilities, renormalized. Every model is trained with balanced
class weights, so its implied class prior is uniform.

--em ALPHA estimates the test class mix from the unlabeled test predictions with the
Saerens-Latinne-Decaestecker EM (no labels involved), then multiplies the posteriors by
(test prior / uniform) ** ALPHA before the argmax. ALPHA 0 is off, 1 is the full Bayes shift.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sig.data import DATA, LABELS, ROOT, load_split  # noqa: E402


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--em", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    weights = args.weights or [1 / len(args.models)] * len(args.models)

    y = load_split("train").y
    _, oof = blend(args.models, weights, "oof")
    ok = ~np.isnan(oof).any(1)
    print(f"OOF macro-F1 (degraded train, {ok.sum()} rows): "
          f"{f1_score(y[ok], oof[ok].argmax(1), average='macro'):.4f}")

    ids, p = blend(args.models, weights, "test")
    prior = em_prior(p)
    print("EM test prior:", dict(zip(LABELS, prior.round(3))))
    if args.em:
        p = p * (prior * len(LABELS)) ** args.em
    pred = np.array(LABELS)[p.argmax(1)]

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
