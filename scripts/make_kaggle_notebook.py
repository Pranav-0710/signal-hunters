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

Steps: folds -> features (train, PKS-degraded train, test) -> adversarial check -> LightGBM ->
first submission -> CNN on the global-time image -> blended submission. All outputs land in
`/kaggle/working/signal-hunters/submissions` and `/kaggle/working/signal-hunters/oof`."""

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
    ("md", "## 1. Folds and features (about 3-5 minutes)"),
    ("code", "!python scripts/make_folds.py\n!python scripts/build_features.py"),
    ("md", "## 2. Does degraded train look like test? (adversarial AUC, 0.5 = identical)"),
    ("code", "!python scripts/adversarial.py --train train\n!python scripts/adversarial.py --train train_deg"),
    ("md", "## 3. LightGBM and the first submission\n\nDownload `submissions/01_lgbm.csv` from the output panel when this cell ends."),
    ("code", "!python scripts/train_lgbm.py --name lgbm\n!python scripts/make_submission.py --models lgbm --out 01_lgbm"),
    ("md", "## 4. CNN on the global-time image (GPU, about 30-45 minutes)"),
    ("code", "!python scripts/train_cnn.py --name cnn"),
    ("md", "## 5. Blends"),
    ("code", "!python scripts/make_submission.py --models cnn --out cnn_only\n"
             "!python scripts/make_submission.py --models lgbm cnn --weights 0.5 0.5 --out 02_blend\n"
             "!python scripts/make_submission.py --models lgbm cnn --weights 0.5 0.5 --em 0.5 --out 03_blend_em"),
    ("md", "## 6. Pack everything for download"),
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
