# Signal Hunters - Team Cyclops

Hack4Hack ML Challenge 2026, Round 2. Each sample is a SETI ON/OFF cadence (6 panels x 16 time
steps x 128 frequency channels); the task is to classify it into one of 9 labels, scored by
macro-F1. Train comes from one telescope (GBT), test from another (PKS). The full task is in
`docs/PROBLEM_STATEMENT.md`; every upload, its score and the exact command behind it are in
`docs/SUBMISSIONS.md`.

## Result

| | Local OOF macro-F1 (PKS-like train) | Public leaderboard |
|---|---|---|
| Final: stack of 11 models + per-class bias (`22_stackall_bias.csv`) | 0.9025 | **0.9067** |
| Main-run stack of 8 models + bias (`18_stack7_bias.csv`) | 0.8998 | 0.9044 |
| First stack with a pretrained CNN (`16_stack6_bias.csv`) | 0.8924 | 0.8948 |
| Stack without pretrained CNNs (`11_stack5_bias.csv`) | 0.8917 | 0.8865 |
| LightGBM alone (`01_lgbm.csv`) | 0.7985 | 0.7884 |

## Approach

**1. Make the training telescope look like the test telescope.** PKS data is noisier than GBT, drops
more rows and has larger gain changes between panels. `src/sig/prep.py::degrade` multiplies the
linear power of each training cadence by Gamma(7.7)/7.7 noise, drops 0.23% more rows and adds a
per-panel gain jitter (sd 0.062 in log2 power). These numbers were measured on the unlabelled test
set and reproduce its noise level exactly (byte MAD 18.5). Every model trains on original plus
degraded cadences and is validated on the degraded copy, which tracks the public score within
about 0.005. Telescope metadata (`telescope`, `tsamp_s`, `channel_width_hz`, `center_freq_mhz`)
is never used as a feature.

**2. Telescope-neutral normalization** (`prep.py::normalize`): dropped rows become missing, each
row's median is removed (gain changes, broadband glitches), a smooth bandpass is subtracted, the
GBT centre-channel spike is removed, and the cadence is scaled by its robust noise level.

**3. Features + LightGBM** (`src/sig/features.py`, `scripts/train_lgbm.py`; 426 features):
per-panel drift search, a global-time ETI track search (ON-panel minimum, OFF-panel penalty,
coherent sum, through the slew gaps), a sub-channel line fit that separates connected ETI tracks
from "same drift but disconnected" traps, dispersed-sweep search for pulsars, FRBs and radar,
texture statistics for scintillation, and panel brightness levels (a scintillating star lifts the
whole ON panel, which row normalization would otherwise erase).

**4. CNNs** (`scripts/train_cnn.py`) on the global-time image: panel p, row r is drawn at time
`p * (16 + gap) + r`, so a real ETI signal is one straight line through the three ON panels.
Channels: normalized power, an ON/OFF mask and the panel brightness. Fresh PKS degradation every
epoch, frequency-flip augmentation and flip TTA.
- three small CNNs trained from scratch (OOF 0.818 - 0.836);
- ImageNet-pretrained timm backbones: EfficientNet-B0 (12 and 20 epochs), EfficientNet-B2,
  ResNet-34, and an EfficientNet-B0 that sees the six panels as channels (OOF 0.80 - 0.87).
  Pretrained weights are public and allowed by the rules; they were the largest single gain on the
  test telescope (+0.008 public from one model).

**5. Stacking** (`scripts/stack.py`): a second-level LightGBM on every model's out-of-fold
log-probabilities plus the line-fit, track and brightness features, trained on the same 5 folds.

**6. Per-class bias** (`scripts/make_submission.py --bias`): one log-probability offset per class,
tuned by coordinate ascent to maximize OOF macro-F1; the gain is measured on folds the offsets were
not tuned on (+0.002 to +0.005 OOF, +0.008 public).

### Tried and dropped
- Test class-mix correction from an EM estimate of the test prior: 0.8806 vs 0.8865 public.
- A stacker on all 426 raw features: 0.8795 public (overfits to GBT).
- Additive-power, patchiness and twinkle features for faint scintillation: no signal beyond
  panel brightness (AUC 0.52 - 0.57 on the confused cases).

**7. Pseudo-labelling** (`train_lgbm.py --pseudo`, `stack.py --pseudo`): test cadences that the
11-model stack labels with at least 90% confidence are added, with that label, to every fold's
training set of LightGBM and of the stacker, so both see real PKS cadences. Validation stays on
labelled training rows. No test cadence is ever labelled by hand. Final file:
`submissions/26_stackpl_bias.csv` (cells at the end of `signal_hunters_kaggle_final.ipynb`).

## Reproducing the final upload

All training runs on Kaggle (GPU T4). The notebooks in `notebooks/` are generated from this repo
by `python scripts/make_kaggle_notebook.py` and write every source file verbatim, so they run
exactly the committed code. Add the Round 2 data as an input to each.

1. `signal_hunters_kaggle.ipynb` (GPU, internet on for the timm weights, about 9 hours): folds,
   features, LightGBM, three small CNNs, four pretrained CNNs, stacks, `18_stack7_bias.csv`.
2. `signal_hunters_kaggle_extra.ipynb` (GPU, internet on, about 4 hours): three more pretrained
   CNNs (same folds).
3. `signal_hunters_kaggle_final.ipynb` (CPU, a few minutes): inputs are the outputs of 1 and 2;
   stacks all 11 models and writes `submissions/22_stackall_bias.csv`.

Seeds are fixed (42 for folds, features and LightGBM; per-model seeds for the CNNs). GPU training
is not bit-for-bit deterministic, so a rerun matches the scores to within about 0.002: a clean
rerun of upload 04 gave OOF 0.8866 against the original 0.8876.

Locally (CPU, slow): `pip install -r requirements.txt`, put the data in `data/`, then run the
commands in the notebook cells in order (`make_folds.py`, `build_features.py --gain-jitter`,
`train_lgbm.py`, `train_cnn.py ...`, `stack.py ...`, `make_submission.py ...`).

## Layout

```
src/sig/data.py        paths, labels, loaders, folds
src/sig/prep.py        PKS degradation and normalization
src/sig/features.py    cadence features
scripts/               folds, features, LightGBM, CNN, stack, submission, notebook builder
notebooks/             the three Kaggle notebooks
docs/SUBMISSIONS.md    every upload with its command and score
```
