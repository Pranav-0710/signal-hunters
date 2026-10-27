# Upload log

Round 2 allows 5 uploads in total. Every upload is listed with the commit and the exact command that rebuilds it.
"OOF" is macro-F1 on the PKS-degraded training copy, 5 folds from `folds.csv`.

| # | File | Commit | Command | OOF | Public | Notes |
|---|---|---|---|---|---|---|
| 01 | `01_lgbm.csv` | `d0ea2a8` | Kaggle notebook: `train_lgbm.py --name lgbm`, `make_submission.py --models lgbm --out 01_lgbm` | 0.798 | **0.7884** | LightGBM on 243 features, original + degraded train. Uploaded 10:17 IST, 1st of 4 at 10:30 |
| 02 | `02_blend.csv` | `d0ea2a8` | Kaggle notebook: `train_cnn.py --name cnn`, `make_submission.py --models lgbm cnn --weights 0.5 0.5 --out 02_blend` | 0.853 | **0.8402** | Log-prob blend of LightGBM (0.798) and global-time CNN (0.817). Uploaded 10:56 IST, 1st of 4 |
