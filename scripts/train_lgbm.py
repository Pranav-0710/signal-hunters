"""LightGBM on the hand-built features, 5 folds from folds.csv.

Trains on original + PKS-degraded train rows, early-stops on the degraded validation fold (the
closest thing to test we have), and writes:
    oof/<name>_oof.csv       degraded-train OOF probabilities (used for blending and tuning)
    oof/<name>_oof_orig.csv  original-train OOF probabilities
    oof/<name>_test.csv      test probabilities, mean over folds
    python scripts/train_lgbm.py --name lgbm
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
    objective="multiclass", num_class=len(LABELS), learning_rate=0.05, num_leaves=31,
    min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, n_jobs=CPUS, verbose=-1, seed=SEED,
)


def save(path: Path, ids: pd.Series, p: np.ndarray) -> None:
    pd.concat([ids.reset_index(drop=True), pd.DataFrame(p, columns=LABELS)], axis=1).to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lgbm")
    ap.add_argument("--no-orig", action="store_true", help="train on degraded rows only")
    args = ap.parse_args()

    tr, te = load_split("train"), load_split("test")
    f_tr = pd.read_parquet(FEATURES / "train.parquet")
    f_dg = pd.read_parquet(FEATURES / "train_deg.parquet")
    f_te = pd.read_parquet(FEATURES / "test.parquet")
    y, folds = tr.y, load_folds()
    n = len(y)

    oof_dg = np.zeros((n, len(LABELS)))
    oof_tr = np.zeros((n, len(LABELS)))
    test = np.zeros((len(f_te), len(LABELS)))
    for k in range(5):
        fit = folds != k
        xs = [f_dg[fit]] if args.no_orig else [f_tr[fit], f_dg[fit]]
        x_fit = pd.concat(xs, ignore_index=True)
        y_fit = np.concatenate([y[fit]] * len(xs))
        d_fit = lgb.Dataset(x_fit, y_fit, weight=compute_sample_weight("balanced", y_fit))
        d_val = lgb.Dataset(f_dg[~fit], y[~fit], weight=compute_sample_weight("balanced", y[~fit]))
        m = lgb.train(PARAMS, d_fit, 3000, valid_sets=[d_val],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_dg[~fit] = m.predict(f_dg[~fit], num_iteration=m.best_iteration)
        oof_tr[~fit] = m.predict(f_tr[~fit], num_iteration=m.best_iteration)
        test += m.predict(f_te, num_iteration=m.best_iteration) / 5
        print(f"fold {k}: iters {m.best_iteration}  degraded F1 "
              f"{f1_score(y[~fit], oof_dg[~fit].argmax(1), average='macro'):.4f}", flush=True)

    print(f"OOF macro-F1  original {f1_score(y, oof_tr.argmax(1), average='macro'):.4f}  "
          f"degraded {f1_score(y, oof_dg.argmax(1), average='macro'):.4f}")
    print(classification_report(y, oof_dg.argmax(1), target_names=LABELS, digits=3))
    print("test predicted mix:", pd.Series(np.array(LABELS)[test.argmax(1)]).value_counts().to_dict())

    imp = pd.Series(m.feature_importance("gain"), index=f_tr.columns).sort_values(ascending=False)
    print("top features:", imp.head(25).round(0).to_dict())
    (ROOT / "oof").mkdir(exist_ok=True)
    save(ROOT / f"oof/{args.name}_oof.csv", tr.meta["id"], oof_dg)
    save(ROOT / f"oof/{args.name}_oof_orig.csv", tr.meta["id"], oof_tr)
    save(ROOT / f"oof/{args.name}_test.csv", te.meta["id"], test)


if __name__ == "__main__":
    main()
