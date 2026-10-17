# `shared_training_pool.py`

Defines one reproducible, balanced development pool for both SL and DL. The
pool is chosen from the existing `dataset_bbox/train` folder only. No train/test
partition is rebuilt, no detector is rerun, and no image file is moved or deleted.

## Shared Image List

The authoritative list is:

`F:/01_Univalle/01_TG/training_protocol/balanced_training_pool_seed_42.csv`

The companion JSON records the seed, sampling protocol, source image order,
per-species counts and a SHA-256 `training_pool_id`. This is a persistent
reproducibility artifact, not a disposable temporary file. Keep both files with
your experiment records. The ID fingerprints ordered image identities and the
protocol; it is not a hash of image pixels.

CSV columns:
- `pool_order`: position in the shared shuffled development pool.
- `source_index`: position in the original DL ImageFolder index.
- `sample_key`: exact species/original-filename identity.
- `species`: class label.

The selector uses the same ImageFolder ordering and NumPy RNG sequence as the
original DL code: undersample each species to the smallest class, then shuffle
the concatenated selected indices with seed 42. Current data gives 95 images
per species, 1,140 total, selected from 1,235 available training images.

SL does not independently sample its CSV rows. It looks up descriptor rows by
`species/orig_filename` and reorders them using the shared list. Shuffling the
input CSV therefore cannot change the chosen images or their order. Missing
selected descriptors, duplicate training-image keys, changed source order/seed
or edited manifests fail clearly instead of silently choosing another cohort.

## Where It Applies

- Both DL trainers: final-model development pool and outer-CV development pool.
- SL feature selection: filters, wrappers, embedded methods and diagnostics.
- Exported SL datasets: 1,140 original train rows, 2,280 shared augmented views
  and 149 untouched test rows (3,569 total). The original full-feature CSV is
  not overwritten by filtering. See [Shared Augmentation](shared_augmentation.md).
- SL tuning, internal validation and final refit: always the same pool.
- SL nested CV: same ordered development population, with selection inside folds.

The selection manifest now uses `train_only_v4_shared_augmentation` and records
`training_pool_id`. Earlier unbalanced exports, including v2 Fourier-aware
exports, must be regenerated. Batch training checks the recorded ID against the
current pool before starting. The augmentation migration requires re-extracting
descriptors and regenerating selected CSVs and models.

The shared list is copied into `sl_results/training_pool.csv`, each SL run's
`metadata/training_pool.csv`, and each new DL final/CV output folder's
`training_pool.csv`, with JSON provenance alongside it. New DL final output also
has `split_membership.csv` showing actual internal train/validation assignments.
The final checkpoint analysis and outer-CV rows record the pool ID.

## Validation Is Still Internal

The shared pool is the population *before* internal train/validation splitting.
These procedures intentionally remain unchanged:

| Procedure | Internal fitting | Internal validation | Final model |
| --- | --- | --- | --- |
| SL, 20% validation | 912 originals + 1,824 views | 228 originals | Refit on 1,140 originals + 2,280 views |
| DL, 15% validation | 969 originals + 1,938 views | 171 originals | Best internal-validation checkpoint; no full-pool refit |

SL no longer performs a second validation-balancing operation when the shared
pool is enabled. At the default 20% fraction its validation is already exactly
balanced. A different fraction can require integer rounding; the entire
stratified validation subset is retained rather than undersampled again.
Existing balanced test-metric tables and full-test image-error audits are unchanged.

Validation is folded back into final fitting for SL, **not** for the existing DL
training procedure. Shared pool membership does not mean identical gradient-fit
or internal-validation membership across families. Matching those as well would
be a separate protocol change; this update does not make it.

## Run Order

```powershell
python 01_Python/run_pipeline.py
```

The runner verifies the shared list, generates shared augmented views, extracts
descriptors, retrains both families and regenerates independent CV and reports.
A list-only check, without augmentation or training, is:

```powershell
python 01_Python/shared_training_pool.py
```

Do not reuse SL rankings/models fitted before this pool change. Restart any
old-protocol run that was already executing. Recreate SL nested-CV statistics if
used for thesis evidence. Splitting, cropping and descriptor extraction do not
need rerunning solely for balancing.

The DL balancing algorithm and selected order have not changed. Existing DL
checkpoints can be reused if they used the same source images/order, seed and
undersampling settings. Historical checkpoints without saved memberships cannot
be retroactively certified from weights alone; new runs save explicit evidence.
For balancing alone this remains true; however the current augmentation migration
changes the DL training policy and requires DL retraining. The runner now defaults
to a full rebuild. See [Shared Augmentation](shared_augmentation.md).

Do not edit the list or silently overwrite it after changing the dataset. Keep
the old list with its experiments and establish a new version before regenerating
selection and models. For isolated library tests only, `pool_manifest_path=None`
can explicitly bypass project matching; normal script entry points use the shared pool.
