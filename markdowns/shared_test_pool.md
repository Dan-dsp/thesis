# Matched SL and DL Test Images

`shared_test_pool.py` reproduces SL's seeded per-species undersampling against test metadata in `dataset_features/shallow_learning_birds.csv`. It maps the chosen `species/orig_filename` identities into each caller's order, so differing CSV/ImageFolder order cannot silently change the cohort. Missing or duplicate identities fail explicitly.

Both DL trainers, shared-pool SL training and `compare_dl_sl_models.py` use this helper. Comparison now defaults to balanced testing; `--no-balance-test` explicitly requests all test images. Full prediction/error audits still cover all 149 images.

Run `python audit_shared_test_pool.py` from `01_Python` for a read-only data audit (it writes reports, not dataset changes). Results are under `training_protocol/test_cohort_audit/` at the project root.

The audit on 2026-11-03 found that the legacy SL and DL selectors already chose exactly the same 132 of 149 crops, with 132 in their intersection and zero unique to either family. New matched manifests also agree in order. The JSON and SL/DL CSV manifests provide the actual identities, not merely equal counts. Each class contributes 11 test images.

Original images and train/test folders are untouched. Matching only selects evaluation indices. `balanced_test_pool.json` records a hash of the ordered selected identities.

The same audit checked all 29 existing exported training CSVs: every legacy selected test set and every newly matched ordered cohort agreed with the reference. Per-export evidence is recorded in `identity_check.json`.
