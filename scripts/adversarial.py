"""Adversarial validation: how well can LightGBM tell train features from test features?

AUC 0.5 means the two look the same. Prints the AUC and the features that give the telescope away.
    python scripts/adversarial.py --train train_deg
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sig.data import CPUS, FEATURES, SEED  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_deg")
    ap.add_argument("--drop", nargs="*", default=[])
    args = ap.parse_args()
    a = pd.read_parquet(FEATURES / f"{args.train}.parquet").drop(columns=args.drop)
    b = pd.read_parquet(FEATURES / "test.parquet").drop(columns=args.drop)
    x = pd.concat([a, b], ignore_index=True)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    p = np.zeros(len(y))
    imp = 0
    for fit, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(x, y):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=CPUS, verbose=-1)
        m.fit(x.iloc[fit], y[fit])
        p[va] = m.predict_proba(x.iloc[va])[:, 1]
        imp = imp + m.booster_.feature_importance("gain")
    print(f"adversarial AUC {args.train} vs test: {roc_auc_score(y, p):.3f}")
    print(pd.Series(imp, index=x.columns).sort_values(ascending=False).head(20).round(0).to_string())


if __name__ == "__main__":
    main()
