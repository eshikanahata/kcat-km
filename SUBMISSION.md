# Submission s16_ens_lf_esm8_pub_tuned

Built 2026-11-01 13:12 (code commit 24832d5-dirty).

## Models
* first model: run `20261101-121510_ens_lf_esm8_pub_tuned` (config in records/runs/20261101-121510_ens_lf_esm8_pub_tuned/config.yaml)
* second model: run `20261101-111423_lgbm_lf_l6_pub`, blended with per-target weights on the second model {'log10_kcat': 0.55, 'log10_km': 0.55}

## Cross-validated score of the submitted blend (5-fold, enzyme-cluster split)

* mean_scc: 0.5884
* kcat_scc: 0.4851
* km_scc: 0.6917
* mean_rmse: 1.1300

## Extra training data
records/data/extra_public.parquet: rows used for TRAINING ONLY (never scored); see scripts/import_extra.py and scripts/source_overlap.py for how they were built and checked.

## Rebuild

```bash
python scripts/run.py configs/ens_lf_esm8_pub_tuned.yaml --no-val  # (tuned runs: then python scripts/tune_weights.py <name>)
python scripts/run.py configs/lgbm_lf_l6_pub.yaml --no-val  # (tuned runs: then python scripts/tune_weights.py <name>)
python scripts/build_submission.py 20261101-121510_ens_lf_esm8_pub_tuned --second 20261101-111423_lgbm_lf_l6_pub --compiled --docker
```

Inference entry point: `python predict.py --input INPUT.parquet --output OUTPUT.csv` (code/submission_template/predict.py; staged, fail-safe, 10-minute budget on 2 CPUs).
