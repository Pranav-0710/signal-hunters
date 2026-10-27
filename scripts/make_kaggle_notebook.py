"""Build notebooks/signal_hunters_kaggle.ipynb from the repo code.

Every source file goes into the notebook verbatim through %%writefile, so the notebook runs
exactly the code in this repo. Rerun this script after any change to src/ or scripts/.
    python scripts/make_kaggle_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "src/sig/data.py", "src/sig/prep.py", "src/sig/features.py",
    "scripts/make_folds.py", "scripts/build_features.py", "scripts/adversarial.py",
    "scripts/train_lgbm.py", "scripts/train_cnn.py", "scripts/make_submission.py", "scripts/stack.py",
]

INTRO = """# Signal Hunters - Team Cyclops (Hack4Hack 2026 Round 2)

Runs the code from the team repository end to end on Kaggle.

**Before running:** add the Round 2 data as an input dataset (any name; the notebook finds
`train_cadences.npy`), and set **Accelerator: GPU T4** for the CNN. Internet is not needed.

Steps: folds -> features (train, PKS-degraded train with gain jitter, test) -> LightGBM ->
three CNNs on the global-time image -> second-level stack -> per-class bias -> final submission
`submissions/11_stack5_bias.csv`. All outputs land in `/kaggle/working/signal-hunters/submissions`
and `/kaggle/working/signal-hunters/oof`."""

SETUP = """import glob, os
hits = glob.glob('/kaggle/input/**/train_cadences.npy', recursive=True)
assert hits, 'Add the Round 2 data as an input dataset first'
os.environ['SIG_DATA'] = os.path.dirname(hits[0])
os.environ['SIG_FEATURES'] = '/kaggle/working/features'
for d in ['src/sig', 'scripts', 'oof', 'submissions']:
    os.makedirs(f'/kaggle/working/signal-hunters/{d}', exist_ok=True)
os.chdir('/kaggle/working/signal-hunters')
print(os.environ['SIG_DATA'], sorted(os.listdir(os.environ['SIG_DATA'])))
!nvidia-smi -L; nproc"""

RUNS = [
    ("md", "## 1. Folds and features (train, PKS-degraded train with gain jitter, test)"),
    ("code", "!python scripts/make_folds.py\n!python scripts/build_features.py --gain-jitter"),
    ("md", "## 2. LightGBM on original + degraded train"),
    ("code", "!python scripts/train_lgbm.py --name lgbm4"),
    ("md", "## 3. CNNs on the global-time image (GPU, about 1 hour each)"),
    ("code", "!python scripts/train_cnn.py --name cnn"),
    ("code", "!python scripts/train_cnn.py --name cnn_s7 --seed 7 --epochs 18"),
    ("code", "!python scripts/train_cnn.py --name cnn_lvl --levels --epochs 18"),
    ("md", "## 4. Stack, per-class bias and the final submission"),
    ("code", "!python scripts/stack.py --models lgbm4 cnn cnn_s7 cnn_lvl --features fit_ trk_ lvl_ --name stack5\n"
             "!python scripts/make_submission.py --models stack5 --bias --out 11_stack5_bias"),
    ("md", "## 5. Pack everything for download"),
    ("code", "!cd /kaggle/working && zip -qr outputs.zip signal-hunters/oof signal-hunters/submissions && ls -la /kaggle/working"),
]


def cell(kind: str, text: str) -> dict:
    src = text.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def main() -> None:
    cells = [cell("md", INTRO), cell("code", SETUP), cell("md", "## 0. Repository code (written verbatim)")]
    for f in FILES:
        cells.append(cell("code", f"%%writefile {f}\n" + (ROOT / f).read_text(encoding="utf-8")))
    cells += [cell(k, t) for k, t in RUNS]
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = ROOT / "notebooks/signal_hunters_kaggle.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} with {len(cells)} cells")


if __name__ == "__main__":
    main()
