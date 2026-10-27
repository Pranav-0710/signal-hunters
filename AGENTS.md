# AGENTS.md - Instructions for coding agents

Every coding agent working in this repo follows these rules, whatever tool it is. Humans follow them too.

## Project context

Team Cyclops, ML Challenge Hack4Hack 2026 Round 2 (Signal Hunters). Classify each of 10,000 test ON/OFF radio cadences (6 panels x 16 time steps x 128 channels) into one of 9 labels. Metric: macro-F1. Round closes 2026-10-28 10:00 AM IST (extended from 27 Oct 11:59 PM; 10 uploads in total). Read `docs/PROBLEM_STATEMENT.md` first.

Stack: Python, numpy, pandas, scipy, scikit-learn, LightGBM, PyTorch (CPU).

Owner: Jey Pranav owns every issue. Goutam Adityan is on the team but not available for this round.

## Git workflow

- Confirm the branch before starting: `git branch --show-current`
- Work only on the branch named in the task; never switch branches mid-task
- Branch names: `type/issue-number-short-description` (e.g. `feat/3-drift-features`)
- Commit format: `type: short description`, under 72 characters; types: feat, fix, chore, docs, refactor, test
- Stage only files relevant to the task; never `git add .` or `git add -A`
- No co-author trailers and no tool attribution in commits, PRs, issues, comments or files
- Never push to `main`, never merge PRs, never force-push a shared branch
- Never close or edit issues, or comment on PRs, on someone's behalf without their confirmation
- Never create or modify `.github/workflows/` without explicit instruction
- When done, list every file modified or created

## Competition rules that bind the code

- Train only on the provided data. No outside labelled data. No hand-labelling the test set. No pseudo-labelling.
- Train is from telescope GBT, test from PKS. Never use `tsamp_s`, `channel_width_hz`, `center_freq_mhz` or `telescope` as features: they identify the telescope, not the signal. `slew_gap_rows` is allowed (it places panels on the global time axis).
- Byte value 0 means a dropped row; treat it as missing.
- Code must reproduce our best upload exactly: fixed seeds (42), one script per submission, every upload logged in `docs/SUBMISSIONS.md` with its commit hash.

## File scope

- Only modify files directly related to the task
- Shared contract - flag before changing: `folds.csv`, `src/sig/data.py`, the `oof/` CSV format, `scripts/make_submission.py`
- Never delete files without explicit instruction
- Never commit `data/` or any `.npy`
