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

The manifest contains no training/validation membership. Ordinary training now
pools all 1,140 originals and 2,280 views before a seeded, species-stratified
row-level split. Original/augmentation identities do not constrain that split.
SL retains `train_augmented` only as provenance, not as a split exclusion.

- SL ordinary internal fitting: 2,736 rows; validation: 684 mixed rows (20%).
- DL ordinary internal fitting: 2,907 rows; validation: 513 mixed rows (15%).
- SL ordinary GridSearchCV splits fitting rows normally, including views.
- Final SL refit uses all 3,420 rows. DL retains the best internal-validation
  checkpoint and does not refit on its validation rows.
- Test remains the existing, unaugmented dataset and is never added to this pool.

**Methodological warning:** related versions of one source image may cross
ordinary training/validation or search folds. This is augmentation-related
information leakage: those scores can be optimistic and are not independent
source-image performance estimates. The requested policy is recorded as
`pooled_rows_after_augmentation_v1` and printed in training warnings.

The separate independent-CV analysis remains protected: SL nested CV assigns
originals to folds and keeps fitting views away from held-out original sources.
DL outer CV reserves original images and excludes their views from its entire
development pool; only its inner checkpoint-selection split uses pooled rows.
Outer scores therefore retain their separate evaluation role. These safeguards
do not apply to ordinary mixed-row validation scores. Existing original-photo
crop grouping limitations still apply.

Feature selection/ranking remains on the original balanced train rows, avoiding
counting synthetic views as independent evidence for ANOVA/Shapiro diagnostics.
Reduced CSVs retain the selected columns for originals, views and untouched test
images. Augmentation metadata is never a predictor. Export protocol is now
`train_only_v5_pooled_augmented_validation`; old reduced CSVs must be regenerated.

The ordinary split intentionally does not prevent augmented-copy leakage.
It also does not make selection on the entire development pool independent of
internal validation. Nested SL CV
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

## Prediction Identity

Test predictions in both families retain `sample_key = species/orig_filename`,
the image path, true species, predicted species and correctness. The shared error
analysis joins on that key, not CSV position. It can identify images both families
misclassify or only one misclassifies. The generated HTML gallery points to the
original test photographs. Missing predictions remain missing, not correct.

Augmented training images also keep a unique PNG filename and `augmentation_parent`
link in the manifest; this never replaces the original test image ID. Internal
row numbers such as sample_id are not used for cross-family test matching.

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
