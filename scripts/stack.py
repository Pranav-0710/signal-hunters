"""Second-level LightGBM on the base models' OOF probabilities (plus chosen raw features).

Same folds as the base models: the stacker for fold k trains on the other folds' OOF rows and
predicts fold k, so its own OOF score is honest. Test predictions are the mean over the 5 fold
stackers applied to the base models' test probabilities.
    python scripts/stack.py --models lgbm3 cnn cnn_s7 --features fit_ trk_ --name stack
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sig.data import CPUS, FEATURES, LABELS, ROOT, SEED, load_folds, load_split  # noqa: E402

PARAMS = dict(
    objective="multiclass", num_class=len(LABELS), learning_rate=0.03, num_leaves=15,
    min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=5.0, n_jobs=CPUS, verbose=-1, seed=SEED,
)


def inputs(models: list[str], prefixes: list[str], part: str, feat_file: str) -> pd.DataFrame:
    cols = []
    for m in models:
        p = pd.read_csv(ROOT / f"oof/{m}_{part}.csv")[LABELS]
        cols.append(np.log(p.clip(1e-6, 1)).add_prefix(f"{m}__"))
    if prefixes:
        f = pd.read_parquet(FEATURES / feat_file)
        cols.append(f[[c for c in f.columns if c.startswith(tuple(prefixes))]].reset_index(drop=True))
    return pd.concat(cols, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--features", nargs="*", default=[], help="raw feature name prefixes to add")
    ap.add_argument("--name", default="stack")
    ap.add_argument("--pseudo", default=None, help="model whose test probabilities give pseudo-labels")
    ap.add_argument("--pseudo-thr", type=float, default=0.9)
    args = ap.parse_args()

    tr, te = load_split("train"), load_split("test")
    y, folds = tr.y, load_folds()
    x = inputs(args.models, args.features, "oof", "train_deg.parquet")
    x_te = inputs(args.models, args.features, "test", "test.parquet")
    x_ps, y_ps = x_te.iloc[:0], np.zeros(0, dtype=int)
    if args.pseudo:
        p = pd.read_csv(ROOT / f"oof/{args.pseudo}_test.csv")[LABELS].to_numpy()
        keep = p.max(1) >= args.pseudo_thr
        x_ps, y_ps = x_te[keep], p[keep].argmax(1)
        print(f"pseudo-labels: {keep.sum()} test rows at >= {args.pseudo_thr}", flush=True)
    oof = np.zeros((len(y), len(LABELS)))
    test = np.zeros((len(x_te), len(LABELS)))
    for k in range(5):
        fit = folds != k
        x_fit, y_fit = pd.concat([x[fit], x_ps]), np.concatenate([y[fit], y_ps])
        d_fit = lgb.Dataset(x_fit, y_fit, weight=compute_sample_weight("balanced", y_fit))
        d_val = lgb.Dataset(x[~fit], y[~fit], weight=compute_sample_weight("balanced", y[~fit]))
        m = lgb.train(PARAMS, d_fit, 3000, valid_sets=[d_val], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[~fit] = m.predict(x[~fit], num_iteration=m.best_iteration)
        test += m.predict(x_te, num_iteration=m.best_iteration) / 5
        print(f"fold {k}: iters {m.best_iteration}  F1 {f1_score(y[~fit], oof[~fit].argmax(1), average='macro'):.4f}", flush=True)
    print(f"stack OOF macro-F1 (degraded): {f1_score(y, oof.argmax(1), average='macro'):.4f}  inputs {x.shape[1]}")
    print(classification_report(y, oof.argmax(1), target_names=LABELS, digits=3))
    pd.concat([tr.meta[["id"]], pd.DataFrame(oof, columns=LABELS)], axis=1).to_csv(ROOT / f"oof/{args.name}_oof.csv", index=False)
    pd.concat([te.meta[["id"]], pd.DataFrame(test, columns=LABELS)], axis=1).to_csv(ROOT / f"oof/{args.name}_test.csv", index=False)


if __name__ == "__main__":
    main()
