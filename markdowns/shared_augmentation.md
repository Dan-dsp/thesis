# Shared Image Augmentation

`shared_augmentation.py` creates two fixed views of every image in the balanced
DL/SL training pool. It does not preassign training/validation membership, move
original files, modify test images, or rerun the bird detector.

## Images And Provenance

Outputs are under `../training_protocol/augmentation_seed_42/`:

- `images/<species>/<hash>.png`: lossless transformed images used by both families.
- `augmentation_manifest.csv`: original species/filename, saved image path, view,
  flip flag, rotation angle, source/output SHA-256 and augmentation ID.
- `augmentation_protocol.json`: policy, seed, pool ID and manifest checksum.

View 1 applies a horizontal flip and a rotation sampled uniformly from -10 to
10 degrees. View 2 applies only a rotation in that range. Rotations retain the
image size and use bilinear resampling with black fill outside the image.
No hue changes, brightness jitter, vertical flips or random crops are applied.
Inspect these views before accepting the protocol for the thesis; augmentation
is an experimental choice, not a guaranteed improvement.

Parameters derive from seed 42, original image identity/content and view number,
not global RNG state or CSV row order. Rebuilding unchanged sources reproduces
the images. Changed pixels produce a new augmentation ID. Originals stay in
`dataset_bbox`; saved views are separate and never become extra ImageFolder
classes or alter the undersampling population.

Current pool: 1,140 originals plus 2,280 generated views = 3,420 possible fitting
samples. This is still 1,140 source images, not 3,420 independent observations.

## Runtime Splits, Not Preassigned Splits

The augmentation manifest contains no training/validation membership. Existing
splitters choose original images at training time, as before.

- DL: attach the saved views only to the runtime training Subset. Internal
  validation, outer CV validation and test loaders contain originals only.
- SL: extract descriptors from the exact saved PNGs. Store them with
  `split=train_augmented`, `augmentation_parent`, and `augmentation_id` metadata.
  The existing runtime internal split still operates on original `train` rows.
  Add only fitting originals' views to its training matrix.
- SL grid-search and nested CV: `TrainingOnlyAugmentationCV` stratifies originals
  at runtime, expands each fitting fold, and scores original held-out images.
  Augmented relatives cannot enter the opposite side of an original's split.
- Final SL refit uses all 1,140 originals and their 2,280 views. DL retains its
  existing best-checkpoint workflow and does not refit on internal validation.

With current fractions, SL internal fitting uses 912 originals + 1,824 views,
and validation has 228 originals. DL internal fitting uses 969 + 1,938 views,
and validation has 171 originals. The policy and source pool match; the different
validation fractions mean those fitting subsets are not identical.

Feature selection/ranking remains on the original balanced train rows, avoiding
counting synthetic views as independent evidence for ANOVA/Shapiro diagnostics.
Reduced CSVs retain the selected columns for originals, views and untouched test
images. Augmentation metadata is never a predictor. Export protocol is now
`train_only_v4_shared_augmentation`; old reduced CSVs must be regenerated.

This change prevents augmented-copy leakage. It does not fix the separate,
previously discussed original-photo crop grouping issue or make selection on
the entire development pool independent of internal validation. Nested SL CV
fits MI/ANOVA selection inside each inner fold; it does not evaluate all 29
exported feature-selection variants independently.

## Randomness Audit

- Pool undersampling: local NumPy generator, seed 42; selected IDs persisted.
- Augmentation: per-image/view local generator and saved parameters/pixels.
- Internal splits, RFECV/search folds: explicit integer seeds.
- PCA: explicit `random_state=seed`, including randomized SVD.
- MI and permutation importance: explicit seeds, propagated from the workflow.
- Random Forest, XGBoost and SVM probability fitting: estimator seeds provided.
- DL: Python, NumPy, CPU/CUDA PyTorch seeds; per-fold seed offsets; final training
  resets its seed so running CV first does not change its starting RNG state.
  cuDNN benchmarking is disabled and deterministic convolution mode enabled.

This is controlled, best-effort reproducibility, not a promise of bitwise equal
training across GPUs, library releases, BLAS implementations or parallel kernels.
Record package versions and hardware with thesis runs. Fixed-seed reproducibility
also does not measure sensitivity to different seeds; repeated-seed experiments
would be a separate analysis.

## Rebuild

From the project parent folder:

```powershell
python 01_Python/run_pipeline.py
```

This now rebuilds both families and independent CV/statistics by default. It
regenerates shared views, descriptors and selections before retraining. Old DL
models used online random augmentation and are not equivalent to this policy.
Do not reuse their results as if they were regenerated with shared augmentation.
Do not run feature extraction/training from another process while rebuilding
the shared images. Existing source train/test folders are reused, not recreated.
