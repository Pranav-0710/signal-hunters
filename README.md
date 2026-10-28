# Signal Hunters - Team Cyclops

Hack4Hack ML Challenge 2026, Round 2. Classify SETI ON/OFF radio-telescope cadences into 9 classes,
scored by macro-F1. Training data comes from one telescope (GBT) and the test set from another
(PKS), so the core of the problem is building models that survive a change of instrument.

**Best public score: 0.9088** (`27_stackpl2_bias.csv`), up from 0.7884 for the first upload.
Every upload, its score and the exact commands that produced it are listed in
[`docs/SUBMISSIONS.md`](docs/SUBMISSIONS.md). The task itself is in
[`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md).

---

## Contents

1. [The task](#1-the-task)
2. [Results](#2-results)
3. [Pipeline overview](#3-pipeline-overview)
4. [Method in detail](#4-method-in-detail)
5. [What did not work](#5-what-did-not-work)
6. [Reproducing the results](#6-reproducing-the-results)
7. [Repository layout](#7-repository-layout)
8. [Rules and compliance](#8-rules-and-compliance)

---

## 1. The task

Each sample is one **cadence**: 6 panels observed in the order ON, OFF, ON, OFF, ON, OFF, each
16 time steps x 128 frequency channels, stored as bytes (`value = round(24 x (log2(power) + 4))`,
0 = dropped row). The telescope slews between panels; panel *p*, row *r* happened at global time
`p x (16 + slew_gap_rows) + r`.

| Label | What it looks like |
|---|---|
| `eti_candidate` | A drifting narrow line in all three ON panels, none of the OFF panels, on one straight track through time (about 3%) |
| `rfi_persistent` | A narrow line in every panel |
| `rfi_intermittent` | Lines in some panels; includes the traps (ON + one OFF, disconnected ON lines, non-drifting ON lines) |
| `satellite_pass` | A bright, steep, sometimes curved line in one or two consecutive panels |
| `pulsar` | Repeating dispersed broadband sweeps, ON panels only |
| `frb` | A single dispersed sweep in one ON panel |
| `radar_chirp` | Repeating steep sawtooth sweeps in almost every panel |
| `scintillating_star` | Patchy broadband brightening in the ON panels only |
| `noise` | Nothing but noise and instrument effects |

A cadence with several sources takes the label of the most important one. The two hard parts:

- **Telescope shift.** PKS is noisier (byte MAD 18.5 against 13.3 for GBT), drops more rows
  (0.38% vs 0.15%), has a different bandpass, no centre-channel spike and larger gain changes
  between panels.
- **Rare and look-alike classes.** ETI candidates are about 3% of the data but weigh 1/9 of the
  score, and faint scintillating stars are close to indistinguishable from noise.

## 2. Results

### Public leaderboard (30% of test)

| Upload | What changed | Local OOF* | Public |
|---|---|---|---|
| `01_lgbm` | LightGBM on 243 hand-built features | 0.7985 | 0.7884 |
| `02_blend` | + small CNN on the global-time image | 0.853 | 0.8402 |
| `08_stack4` | second-level stack, line-fit and brightness features | 0.887 | 0.8781 |
| `11_stack5_bias` | + brightness-channel CNN, per-class bias | 0.8917 | 0.8865 |
| `16_stack6_bias` | + ImageNet-pretrained EfficientNet-B0 | 0.8924 | 0.8948 |
| `18_stack7_bias` | + EfficientNet-B2, B0 trained 20 epochs, stacked-panel B0 | 0.8998 | 0.9044 |
| `22_stackall_bias` | stack of all 11 models | 0.9025 | 0.9067 |
| **`27_stackpl2_bias`** | final refinement of the 11-model stack (notebook 3, last cells) | **0.9029** | **0.9088** |

\* Out-of-fold macro-F1 on a PKS-like (degraded) copy of the training set, 5 folds. It has tracked
the public score to within about 0.005 all the way through, which is what made it safe to iterate
on locally.

### Single models (OOF, PKS-like train)

| Model | Input | OOF macro-F1 |
|---|---|---|
| LightGBM (`lgbm4`) | 426 features | 0.842 |
| Small CNN (`cnn`, `cnn_s7`) | global-time image | 0.818, 0.824 |
| Small CNN + brightness (`cnn_lvl`) | global-time image + panel levels | 0.836 |
| EfficientNet-B0, 12 epochs (`eff_lvl`, `eff_lvl_s11`) | global-time image, pretrained | 0.838, 0.839 |
| EfficientNet-B0, six panels as channels (`effstk_lvl`, `effstk_lvl_s7`) | stacked panels, pretrained | 0.801, 0.802 |
| ResNet-34 (`res34_lvl`) | global-time image, pretrained | 0.856 |
| EfficientNet-B2 (`effb2_lvl`) | global-time image, pretrained | 0.851 |
| **EfficientNet-B0, 20 epochs (`eff_lvl_e20`)** | global-time image, pretrained | **0.869** |
| **Stack of all 11 + per-class bias** | | **0.9025** |

### Per-class F1 of the final stack (OOF)

| eti_candidate | frb | noise | pulsar | radar_chirp | rfi_intermittent | rfi_persistent | satellite_pass | scintillating_star |
|---|---|---|---|---|---|---|---|---|
| 0.781 | 0.983 | 0.934 | 0.991 | 0.937 | 0.920 | 0.939 | 0.917 | 0.719 |

The remaining errors are concentrated in two pairs: faint scintillating stars vs noise (the star's
brightening is about the size of the telescope's own gain jitter) and ETI candidates vs the
"disconnected lines" interference trap.

## 3. Pipeline overview

```
raw cadences (uint8)
  |-- PKS-style degradation of train (noise, dropped rows, gain jitter)   src/sig/prep.py
  |-- telescope-neutral normalization                                      src/sig/prep.py
  |
  |-- 426 hand-built features  -> LightGBM                                 features.py, train_lgbm.py
  |-- global-time images       -> 3 small CNNs + 7 ImageNet-pretrained     train_cnn.py
  |
  |-- out-of-fold probabilities of all 11 models + line-fit, track and
  |   brightness features      -> second-level LightGBM (stack)            stack.py
  |-- per-class log-probability offsets tuned on OOF macro-F1              make_submission.py --bias
  v
submissions/*.csv
```

All base models share the same 5 folds (`scripts/make_folds.py`: StratifiedGroupKFold by target
star, seed 42), so the stack is trained only on genuine out-of-fold predictions.

## 4. Method in detail

### 4.1 Making train look like the test telescope

`src/sig/prep.py::degrade` turns a GBT cadence into a PKS-like one:

- multiplies linear power by Gamma(7.7)/7.7 noise (what averaging fewer spectra does), which widens
  the byte MAD from 13.3 to exactly the 18.5 measured on the unlabelled test cadences and fades
  signals in the same way;
- drops an extra 0.23% of rows;
- adds a per-panel gain jitter (sd 0.062 in log2 power), matching the spread between OFF-panel
  levels on PKS (0.079 vs 0.049).

Every model trains on original plus degraded cadences (CNNs draw a fresh degradation each epoch)
and is validated on a fixed degraded copy.

### 4.2 Telescope-neutral normalization

`prep.py::normalize`, per cadence: dropped rows become missing; the GBT centre-channel spike is
replaced by its neighbours; each row's median is removed (gain changes and broadband glitches); a
smooth bandpass (running median of the per-channel median) is subtracted, which keeps persistent
narrow lines; the result is scaled by the robust noise level.

### 4.3 Hand-built features (426) and LightGBM

`src/sig/features.py`:

- **Per-panel drift search**: S/N of the best straight line at every (drift, channel), summarized
  per panel and as ON/OFF contrasts; static, slow and steep drift bands; counts of panels with
  lines.
- **Global-time ETI track search**: lines through all six panels on the true time axis (slew gaps
  included), scored by the weakest ON panel, an OFF-panel penalty and a coherent ON sum.
- **Sub-channel line fit** around the best track: per-panel offsets and slopes. A real ETI signal
  has one consistent offset; the "same drift but disconnected" trap does not.
- **Persistent-line tests**: the same channel lit in every panel, allowing one faded panel.
- **Dispersed-sweep search** for pulsars, FRBs and radar: best sweep per panel, slope, curvature and
  number of repeated peaks.
- **Texture statistics** for scintillation: smoothed patch variance and short-lag
  autocorrelations, after clipping narrow lines.
- **Panel brightness levels** from the raw bytes: a scintillating star lifts the whole ON panel,
  which row normalization would otherwise erase.

`scripts/train_lgbm.py` trains a multiclass LightGBM with balanced class weights and early stopping
on the degraded validation fold. Telescope metadata is never used as a feature.

### 4.4 CNNs

`scripts/train_cnn.py` draws each cadence on its **global-time image** (128 x 128): panel p, row r
sits at row `p x (16 + gap) + r`, so a genuine ETI signal is one straight line through the three ON
panels. Input channels: normalized power, an ON/OFF mask and the panel brightness level.

- Three small CNNs trained from scratch (5 conv blocks, mean + max pooling).
- Seven ImageNet-pretrained `timm` backbones (EfficientNet-B0 and B2, ResNet-34), fine-tuned with
  mixed precision at 256 x 256, including a variant that sees the six panels as stacked channels.
- Fresh PKS-style degradation every epoch, frequency-flip augmentation, flip test-time
  augmentation, label smoothing 0.05, class-balanced loss, one-cycle learning rate.

Pretrained backbones were the single largest gain on the test telescope: one EfficientNet-B0 at
roughly the same OOF as our best from-scratch CNN added +0.008 public, and longer training (20
epochs) lifted a single model from 0.838 to 0.869 OOF.

### 4.5 Stacking

`scripts/stack.py` trains a second-level LightGBM (shallow, strongly regularized) on the
out-of-fold log-probabilities of all base models plus the line-fit, track and brightness features,
on the same folds. Test predictions average the five fold stackers.

### 4.6 Per-class bias

`scripts/make_submission.py --bias` adds one log-probability offset per class, found by coordinate
ascent on OOF macro-F1. Macro-F1 rewards trading a little accuracy on large classes for recall on
small ones. The gain is measured honestly: offsets tuned on four folds, scored on the fifth
(+0.002 to +0.005 OOF, +0.008 public the first time it was used).

## 5. What did not work

| Idea | Result |
|---|---|
| Correcting for the test class mix with an EM estimate of the test prior | 0.8806 vs 0.8865 public: the estimate was skewed by the model's own errors |
| Stacker on all 426 raw features | 0.8795 public: overfits to the training telescope |
| Additive-power, patchiness and twinkle features for faint scintillation | No signal beyond panel brightness (AUC 0.52-0.57 on the confused cases) |
| Rescaling for the PKS time and channel resolution | Drift-rate distributions of strong lines already match between telescopes |
| Blending the stack with its strongest base models | 0.8993-0.9024 OOF, never above the stack alone |

## 6. Reproducing the results

All training ran on Kaggle (GPU T4, 30 GPU-hours a week). The notebooks in `notebooks/` are
generated from this repository by `python scripts/make_kaggle_notebook.py` and write every source
file verbatim, so they run exactly the committed code. Add the Round 2 data as an input dataset to
each (the notebooks find `train_cadences.npy` themselves).

| Step | Notebook | Hardware | Time | Produces |
|---|---|---|---|---|
| 1 | `signal_hunters_kaggle.ipynb` | GPU T4, internet on (timm weights) | about 9 h | folds, features, LightGBM, 3 small CNNs, 4 pretrained CNNs, `18_stack7_bias.csv` |
| 2 | `signal_hunters_kaggle_extra.ipynb` | GPU T4, internet on | about 4 h | 3 more pretrained CNNs on the same folds |
| 3 | `signal_hunters_kaggle_final.ipynb` | CPU | 5-25 min | inputs: outputs of 1 and 2; the 11-model stack (`22_stackall_bias.csv`) and, in its last cells, the refined stacks behind uploads 26-29 including the best, `27_stackpl2_bias.csv` |

The exact command lines for every upload are in [`docs/SUBMISSIONS.md`](docs/SUBMISSIONS.md).

Seeds are fixed (42 for folds, features and LightGBM; per-model seeds for the CNNs). GPU training
is not bit-for-bit deterministic, so a rerun matches the scores to within about 0.002: a full clean
rerun of upload 04 gave OOF 0.8866 against the original 0.8876, and the CPU stacking steps reproduce
exactly.

**Locally** (CPU only, slow for the CNNs): `pip install -r requirements.txt`, put the data in
`data/`, then run the commands from the notebook cells in order: `make_folds.py`,
`build_features.py --gain-jitter`, `train_lgbm.py`, `train_cnn.py ...`, `stack.py ...`,
`make_submission.py ...`.

## 7. Repository layout

```
src/sig/data.py              paths, labels, loaders, folds, global time axis
src/sig/prep.py              PKS-style degradation and telescope-neutral normalization
src/sig/features.py          the 426 cadence features
scripts/make_folds.py        5 stratified folds grouped by target star
scripts/build_features.py    features for train, degraded train and test
scripts/train_lgbm.py        LightGBM base model
scripts/train_cnn.py         small and ImageNet-pretrained CNNs
scripts/stack.py             second-level stacker
scripts/make_submission.py   blending, per-class bias, submission checks
scripts/adversarial.py       train-vs-test adversarial validation
scripts/make_kaggle_notebook.py  builds the three Kaggle notebooks from the repo
notebooks/                   the three Kaggle notebooks
docs/PROBLEM_STATEMENT.md    the task
docs/SUBMISSIONS.md          every upload with its commit, commands, OOF and public score
```

## 8. Rules and compliance

- Trained only on the data provided for Round 2; no outside labelled data, and no test cadence was
  labelled by hand.
- Publicly available pretrained weights (ImageNet, via `timm`) are used, as the rules allow.
- No code, predictions or labels were shared with other teams.
- Telescope metadata (`telescope`, `tsamp_s`, `channel_width_hz`, `center_freq_mhz`) is never a
  feature.

Team Cyclops: Jey Pranav (all code, experiments and submissions), Goutam Adityan.
