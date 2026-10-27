"""Write folds.csv: stratified 5-fold split of train, grouped by target star, seed 42."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sig.data import ROOT, SEED, load_split  # noqa: E402


def main() -> None:
    tr = load_split("train")
    folds = pd.Series(-1, index=tr.meta.index)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(skf.split(tr.meta, tr.meta["label"], tr.meta["target"])):
        folds.iloc[va] = k
    out = pd.DataFrame({"id": tr.meta["id"], "fold": folds})
    out.to_csv(ROOT / "folds.csv", index=False)
    print(pd.crosstab(tr.meta["label"], out["fold"]))


if __name__ == "__main__":
    main()
