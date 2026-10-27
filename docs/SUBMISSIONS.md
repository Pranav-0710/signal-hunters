# Upload log

Round 2 allows 5 uploads in total. Every upload is listed with the commit and the exact command that rebuilds it.
"OOF" is macro-F1 on the PKS-degraded training copy, 5 folds from `folds.csv`.

| # | File | Commit | Command | OOF | Public | Notes |
|---|---|---|---|---|---|---|
| 01 | `01_lgbm.csv` | `d0ea2a8` | Kaggle notebook: `train_lgbm.py --name lgbm`, `make_submission.py --models lgbm --out 01_lgbm` | 0.798 | **0.7884** | LightGBM on 243 features, original + degraded train. Uploaded 10:17 IST, 1st of 4 at 10:30 |
| 02 | `02_blend.csv` | `d0ea2a8` | Kaggle notebook: `train_cnn.py --name cnn`, `make_submission.py --models lgbm cnn --weights 0.5 0.5 --out 02_blend` | 0.853 | **0.8402** | Log-prob blend of LightGBM (0.798) and global-time CNN (0.817). Uploaded 10:56 IST, 1st of 4 |
| 03 | `08_stack4.csv` | `221d6cd` | Kaggle session: `build_features.py --gain-jitter`, `train_lgbm.py --name lgbm4`, `stack.py --models lgbm4 cnn cnn_s7 --features fit_ trk_ lvl_ --name stack4`, `make_submission.py --models stack4 --out 08_stack4` | 0.887 | **0.8781** | Stack of LightGBM v4 (ETI line fit, panel brightness, gain jitter) + 2 CNN seeds (`cnn` at `d0ea2a8`, `cnn_s7 --seed 7 --epochs 18`). Uploaded 12:44 IST, 2nd of 14 |
