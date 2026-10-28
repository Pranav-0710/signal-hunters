"""Build notebooks/signal_hunters_kaggle.ipynb (full pipeline) and
notebooks/signal_hunters_kaggle_extra.ipynb (three more pretrained CNNs) from the repo code.

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
`train_cadences.npy`), set **Accelerator: GPU T4** and turn **Internet on** (the pretrained
backbones download public ImageNet weights through timm). A full run takes about 8 hours.

Steps: folds -> features (train, PKS-degraded train with gain jitter, test) -> LightGBM ->
three small CNNs and three ImageNet-pretrained EfficientNets on the global-time image ->
second-level stack -> per-class bias -> submissions. The final file is
`submissions/18_stack7_bias.csv`. All outputs land in `/kaggle/working/signal-hunters/submissions`
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
    ("md", "## 4. Stack of the small models (upload 04)"),
    ("code", "!python scripts/stack.py --models lgbm4 cnn cnn_s7 cnn_lvl --features fit_ trk_ lvl_ --name stack5\n"
             "!python scripts/make_submission.py --models stack5 --bias --out 11_stack5_bias\n"
             "!python scripts/make_submission.py --models stack5 --bias --shift 0.5 --out 14_stack5_bias_shift05"),
    ("md", "## 5. ImageNet-pretrained EfficientNets (GPU, about 70 minutes for B0, 2 hours for B2)"),
    ("code", "!python scripts/train_cnn.py --arch efficientnet_b0 --levels --lr 1e-3 --epochs 12 --name eff_lvl"),
    ("code", "!python scripts/stack.py --models lgbm4 cnn cnn_s7 cnn_lvl eff_lvl --features fit_ trk_ lvl_ --name stack6\n"
             "!python scripts/make_submission.py --models stack6 --bias --out 16_stack6_bias\n"
             "!python scripts/make_submission.py --models stack6 --bias --shift 0.5 --out 17_stack6_bias_shift05"),
    ("code", "!python scripts/train_cnn.py --arch efficientnet_b0 --layout stacked --levels --lr 1e-3 --epochs 12 --name effstk_lvl"),
    ("code", "!python scripts/train_cnn.py --arch efficientnet_b2 --size 224 --levels --lr 1e-3 --epochs 14 --name effb2_lvl"),
    ("code", "!python scripts/train_cnn.py --arch efficientnet_b0 --levels --lr 1e-3 --epochs 20 --seed 7 --name eff_lvl_e20"),
    ("md", "## 6. Final stack of everything (a model whose run failed is left out and reported)"),
    ("code", "import subprocess\n"
             "models = [m for m in ['lgbm4', 'cnn', 'cnn_s7', 'cnn_lvl', 'eff_lvl', 'effstk_lvl', 'effb2_lvl', 'eff_lvl_e20']\n"
             "          if os.path.exists(f'oof/{m}_oof.csv') and os.path.exists(f'oof/{m}_test.csv')]\n"
             "print('stacking', models)\n"
             "subprocess.run(['python', 'scripts/stack.py', '--models', *models, '--features', 'fit_', 'trk_', 'lvl_',\n"
             "                '--name', 'stack7'], check=True)\n"
             "!python scripts/make_submission.py --models stack7 --bias --out 18_stack7_bias\n"
             "!python scripts/make_submission.py --models stack7 --bias --shift 0.5 --out 19_stack7_bias_shift05"),
    ("md", "## 7. Pack everything for download"),
    ("code", "!cd /kaggle/working && zip -qr outputs.zip signal-hunters/oof signal-hunters/submissions && ls -la /kaggle/working"),
]


EXTRA_INTRO = """# Signal Hunters - Team Cyclops: extra pretrained models

Runs alongside the main notebook (same code, same folds) and trains three more ImageNet-pretrained
CNNs. Their `oof/*_oof.csv` and `oof/*_test.csv` are stacked with the main notebook's models in the
final step. Same setup as the main notebook: Round 2 data as input, **GPU T4**, **Internet on**."""

EXTRA_RUNS = [
    ("md", "## 1. Folds (identical to the main notebook: seed 42)"),
    ("code", "!python scripts/make_folds.py"),
    ("md", "## 2. Three more pretrained CNNs (about 70-90 minutes each)"),
    ("code", "!python scripts/train_cnn.py --arch efficientnet_b0 --levels --lr 1e-3 --epochs 12 --seed 11 --name eff_lvl_s11"),
    ("code", "!python scripts/train_cnn.py --arch efficientnet_b0 --layout stacked --levels --lr 1e-3 --epochs 12 --seed 7 --name effstk_lvl_s7"),
    ("code", "!python scripts/train_cnn.py --arch resnet34 --levels --lr 1e-3 --epochs 12 --name res34_lvl"),
    ("code", "!ls -la oof"),
]


FINAL_INTRO = """# Signal Hunters - Team Cyclops: final stack

CPU only, no internet. Inputs: the Round 2 data, the output of the main notebook (features,
LightGBM, CNNs, EfficientNets, repo code) and the output of the extra-models notebook. All base
models' out-of-fold and test probabilities are stacked by the same second-level LightGBM, then the
per-class bias is applied. Final file: `submissions/22_stackall_bias.csv`."""

FINAL_SETUP = """import glob, os, shutil, subprocess
hits = glob.glob('/kaggle/input/**/train_cadences.npy', recursive=True)
os.environ['SIG_DATA'] = os.path.dirname(hits[0])
roots = [os.path.dirname(os.path.dirname(p)) for p in glob.glob('/kaggle/input/**/oof/lgbm4_oof.csv', recursive=True)]
assert roots, 'Add the main notebook output as an input'
main = roots[0]
shutil.copytree(main, '/kaggle/working/signal-hunters', dirs_exist_ok=True)
os.environ['SIG_FEATURES'] = glob.glob('/kaggle/input/**/features/train_deg.parquet', recursive=True)[0].rsplit('/', 1)[0]
for p in glob.glob('/kaggle/input/**/oof/*.csv', recursive=True):
    dst = '/kaggle/working/signal-hunters/oof/' + os.path.basename(p)
    if not os.path.exists(dst):
        shutil.copy(p, dst)
        print('added', os.path.basename(p))
os.chdir('/kaggle/working/signal-hunters')
print(main, os.environ['SIG_FEATURES'])"""

FINAL_RUNS = [
    ("md", "## Stack every base model, then the per-class bias"),
    ("code", "models = [m for m in ['lgbm4', 'cnn', 'cnn_s7', 'cnn_lvl', 'eff_lvl', 'effstk_lvl', 'effb2_lvl', 'eff_lvl_e20',\n"
             "                     'eff_lvl_s11', 'effstk_lvl_s7', 'res34_lvl']\n"
             "          if os.path.exists(f'oof/{m}_oof.csv') and os.path.exists(f'oof/{m}_test.csv')]\n"
             "print('stacking', models)\n"
             "subprocess.run(['python', 'scripts/stack.py', '--models', *models, '--features', 'fit_', 'trk_', 'lvl_',\n"
             "                '--name', 'stackall'], check=True)\n"
             "!python scripts/make_submission.py --models stackall --bias --out 22_stackall_bias\n"
             "!ls -la submissions"),
]


def cell(kind: str, text: str) -> dict:
    src = text.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def write(name: str, intro: str, runs: list, setup: str = SETUP, code: bool = True) -> None:
    cells = [cell("md", intro), cell("code", setup)]
    if code:
        cells.append(cell("md", "## 0. Repository code (written verbatim)"))
        for f in FILES:
            cells.append(cell("code", f"%%writefile {f}\n" + (ROOT / f).read_text(encoding="utf-8")))
    cells += [cell(k, t) for k, t in runs]
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = ROOT / f"notebooks/{name}.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} with {len(cells)} cells")


def main() -> None:
    write("signal_hunters_kaggle", INTRO, RUNS)
    write("signal_hunters_kaggle_extra", EXTRA_INTRO, EXTRA_RUNS)
    write("signal_hunters_kaggle_final", FINAL_INTRO, FINAL_RUNS, setup=FINAL_SETUP, code=False)


if __name__ == "__main__":
    main()
