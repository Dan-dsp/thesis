# Thesis Roadmap: From Prepared Images to Results

Verified against the current Python code on 2026-10-29. This describes the
configured experiment, not proof that all results have already been regenerated.
The accompanying interactive roadmap is `visuals/thesis-roadmap.html`.
Corrections discussed through it do not automatically change the training code.

## The Thesis in One Paragraph

The project classifies 12 bird species using two approaches. Shallow Learning
(SL) receives handcrafted image descriptors, reduces their dimensionality through
feature selection, and trains SVM, k-NN, Random Forest and XGBoost. Deep Learning
(DL) receives the images and adapts an ImageNet-pretrained ResNet18, using either
full or partial fine-tuning. Both families share the same balanced original
training-image list and saved augmented images. Performance is assessed within
each family using cross-validation and on a separate test set. Final reports
also identify which test images cause errors in SL, DL, or both.

## Methodology Proposed Solution for Shallow Learning

These four sections also appear as selectable sections in the interactive roadmap.
They organize the methodology for thesis writing; they do not add execution stages.

### Introduction

The SL solution identifies bird species using handcrafted numerical descriptions
of each image. Its sequence is: prepared crop, descriptor extraction, feature
selection, classifier fitting, and prediction. Feature selection reduces the
dimensionality before training, rather than assuming every measurement improves
classification. Four classifier families are studied: RBF SVM, k-NN, Random
Forest and XGBoost. The many saved models represent feature/hyperparameter
configurations of these four families, not additional algorithms.

### Dataset Construction With Feature Extraction

The starting data are already cropped and separated into train/test folders.
The shared image list selects 95 training originals per species; two saved views
per original expand the 1,140-image balanced pool to 3,420 development images.
Original files and the separate 149-image test set remain unchanged.

Images are resized to 224 x 224 for descriptor extraction. Shape/structure is
represented through Hu and Zernike moments, Fourier descriptors, affine
invariants and log-polar FFT measurements. Color histograms/moments and LBP,
HOG, GLCM, Gabor and edge descriptors supply complementary color, texture and
gradient information. Each CSV row identifies one original or augmented image,
its species and split, and its numerical descriptors. Identity/provenance columns
are metadata, not classifier inputs. Extracting test descriptors is a fixed
transformation; test labels do not participate in feature selection or fitting.

### Feature Comparison and Selection

#### Distribution Examination: What Was Done and Why

Before ranking, `sl_feature_comparison.py` calls `run_shapiro_tests` and
`plot_correlation_matrix` from `sl_feature_comparison_tools.py`. Both examine
the balanced original training rows, excluding augmented and test rows.

**Shapiro-Wilk:** one test per feature **within each species**, at alpha 0.05.
The purpose is to describe departures from a Gaussian distribution, not decide
whether a classifier is allowed to use the feature. Non-finite values are omitted
for this diagnostic; fewer than three finite values produce an unavailable test.
Above 5,000 values, a seed-42 random subsample is used to limit work and avoid the
large-sample p-value accuracy concern noted in the implementation. The current
pool has 95 originals per species and does not require subsampling.

Interpret p < 0.05 as rejection of normality. A p-value at or above 0.05 means
normality was not rejected, not that it was proven. `shapiro_W` is the test
statistic; `gaussian_at_alpha` records the latter decision. Unavailable tests also
receive a false flag, so inspect `p_value` and `notes` rather than assuming every
false flag is evidence against normality. Constant features are also not
testable. Use the explicit `test_status` field. Historical pooled results
(9 not rejected / 304 rejected) do not describe this within-species protocol.

The outputs are `sl_results/exploratory/shapiro/shapiro_normality.csv`,
`shapiro_summary.json` and `shapiro_p_values.png`. The PNG is a distribution of
**test p-values across features**, not a histogram of each feature's raw values.
Tests use unadjusted per-feature/species p-values: this is exploratory characterization,
not a multiple-testing-controlled claim that each feature is or is not Gaussian.

**Interpretation limits:** the new protocol avoids mixing species distributions.
It is not a multivariate normality test or complete validation of
ANOVA's assumptions. In particular, ANOVA significance interpretation also
depends on within-group/residual behavior, variance assumptions and independence;
the within-species Shapiro result alone does not establish these. The code continues to rank
with ANOVA F-scores regardless of this diagnostic.

The highest and lowest mean-Shapiro-W features per descriptor family are plotted
across all species, with common histogram bins and Gaussian references (up to 24 PNGs).
Only features testable in every species are eligible; ties use feature order.
These are selected extremes, not typical family examples. See [normality diagnostics](sl_normality.md)
for output paths and the diagnostics-only rerun command. This does not normalize
or otherwise transform the model inputs.

**Pearson correlations:** the full pairwise matrix describes linear association
between descriptors and helps identify potentially redundant information.
`exploratory/correlation/global_feature_correlation.csv` stores the full matrix;
the PNG displays only the first 40 features for readability. Near-zero correlation
does not prove independence. This diagnostic does not automatically remove
features above a correlation threshold.

**After selection:** `run_final_selection_diagnostics` uses the same original
training rows for consensus top-50, SVM/k-NN RFECV-selected lists and Random
Forest/XGBoost top-50 lists. In `sl_results/diagnostics/`, each list receives its
feature CSV, full subset-correlation CSV, heatmap of up to 40 features, and violin
plots for up to its first 12 features. Violin widths show estimated concentration
of values and the plots mark medians for each species. These plots help inspect
class separation, spread and overlap. They are visual diagnostics, not formal
per-class normality tests or proof that the classifier will generalize.

**Effect on the pipeline:** these checks do not discard non-Gaussian features,
change selection methods or trigger Gaussianizing transformations. The four SL
classifiers do not require Gaussian input features. SVM/k-NN scaling helps with
feature magnitudes but does not make skewed distributions Gaussian. Actual
selection proceeds through the filters, wrappers and embedded methods below;
predictive usefulness is assessed through validation and the independent CV
protocol, not the Shapiro decision.

#### Selection Methods

Rankings are calculated from the 1,140 balanced original training images.
ANOVA, Fisher and mutual information assess relationships with species labels;
PCA loadings rank original features by their contribution to variation. A
consensus combines filter agreement and ranks. SVM/k-NN wrappers use RFECV to
evaluate progressively reduced subsets, with hyperparameter searches before and
after selection. Random Forest/XGBoost provide embedded importance rankings.

The 29 exported variants comprise 12 filter lists, 3 consensus lists, 8 wrapper
lists and 6 embedded lists. They change feature columns, not the shared image
population. Rankings, overlap tables, importance plots and RFECV curves describe
these choices; downstream validation evaluates classification usefulness.
The exported rankings precede the ordinary batch split, so batch validation is
not independent evidence for the entire selection procedure. Separate nested CV
repeats selection within folds for MI top-50 only.

### Classifier Training and Model Comparison

For each variant, 80% of the augmented development rows are used for fitting and
20% for internal validation. Five-fold GridSearchCV tunes each classifier using
macro-F1; SVM and k-NN learn scaling inside their pipelines. Internal validation
selects the preferred configuration. Each tuned model is then refitted on all
3,420 development rows and saved, giving 116 final models when all four families
are available. This SL process has no neural-network epochs, warmup or fine-tuning.

The independent MI-top-50 experiment uses five outer and five inner folds, with
held-out original images and their views excluded from fitting. Four SL families
yield six paired statistical comparisons using corrected t-tests and Holm
adjustment. This evidence concerns that experiment, not every exported variant.
Final reports include test metrics, confusion matrices, settings, timings and
image-level errors. The shared SL/DL report scores 132 matched test images, audits all 149, and
loads the four SL models from the winning feature-variant run. It describes
cross-family errors without performing a joint SL-versus-DL significance test.

## Start Here

From `F:\01_Univalle\01_TG`:

```powershell
# Inspect the actual execution plan without training or changing results:
python 01_Python/run_pipeline.py --dry-run

# Rebuild the configured experiment, including independent CV and statistics:
python 01_Python/run_pipeline.py

# Reuse saved models; regenerate family reports and the error gallery only:
python 01_Python/run_pipeline.py --reports-only
```

The runner starts each script inside `01_Python`, sequentially, and stops on the
first failure. It does not automatically resume a failed run.
`--skip-statistics` omits stages 8-10 below, but does NOT disable the CV built
into the DL trainers. Merely opening this roadmap runs no experiments.

## Dataset: What "Full" Means

The runner begins with already prepared bird crops in `dataset_bbox/train` and
`dataset_bbox/test`. It does not run the detector, rename images, or create the
original train/test split. These are upstream preparation steps, not extra
stages silently performed by this command.

| Dataset level | Current count | Meaning |
| --- | ---: | --- |
| Available original training images | 1,235 | Files currently in `dataset_bbox/train` |
| Shared balanced originals | 1,140 | 95 per species, selected with seed 42 |
| Saved augmented images | 2,280 | Two versions of each selected original |
| Full balanced development pool | 3,420 | 1,140 originals + 2,280 saved views |
| Original test images | 149 | Separate folder, never augmented or fitted |

"Full training pool" means the 3,420 development rows, not the test set and not
all 1,235 available originals. The 95 surplus original training images are not
used in the balanced experiment. The original image files remain unchanged.
Counts of available files were checked directly; derived counts assume the
current two-view augmentation configuration completes successfully.

## Execution Roadmap

### 1. Choose the Same Original Images for SL and DL

**Script:** `shared_training_pool.py`

Undersample the original training classes to 95 images each, using seed 42.
Persist an ordered shared list so SL selects descriptors for exactly the images
used by DL, rather than making its own unrelated random selection.

**Outputs:** `training_protocol/balanced_training_pool_seed_42.json` and its
CSV companion. They record identities, ordering and the pool identifier.

### 2. Generate Shared Augmented Images

**Script:** `shared_augmentation.py`

Each of the 1,140 originals has two separately saved PNG views:

- Version 1: horizontal flip, then rotation sampled within -10 to +10 degrees.
- Version 2: rotation sampled independently within the same range; no flip.

Angles vary between images/views, but are deterministic for the same seed,
source content and protocol. They are not sampled again each epoch. Rotation
uses bilinear interpolation, retains image dimensions and fills corners black.
There is no additional random brightness, color jitter or vertical flip here.
The original image is still included, making three rows per selected source.

**Outputs:** `training_protocol/augmentation_seed_42/` contains
`augmentation_manifest.csv`, `augmentation_protocol.json` and `images/`.
The manifest connects every view to its original and records its transformation.

### 3. Extract SL Descriptors

**Script:** `sl_dataframe_main.py` (uses `sl_dataframe_construction.py` and `sl_methods.py`)

Read original train/test crops and the saved augmented views, resize to 224 x
224 for extraction, and compute handcrafted descriptors. No classifier is
trained at this stage. DL does not consume these descriptor CSVs.

**Outputs:** `dataset_features/shallow_learning_birds.csv`,
`features_manifest.csv` and per-image feature CSVs. With the current inputs the
master contains 3,664 rows: 1,235 available train originals + 2,280 augmented
views + 149 test originals. Downstream selection enforces the shared balanced
pool; the extra original training rows are not all used for training.

### 4. Train DL: Full Fine-Tuning

**Script:** `dl_transfer_learning_bird_classifier.py`

First run five-fold outer CV. Then initialize a fresh model for final training.
Both use pretrained ResNet18, resized/normalized images, batch size 32, Adam and
cross-entropy loss. CUDA is used when available; otherwise CPU.

| Training run | Repetitions | Warmup | Fine-tuning | Internal validation |
| --- | ---: | ---: | ---: | --- |
| Outer-CV model | 5, one per fold | 2 epochs | 3 epochs | 20% of the outer development pool |
| Final selection | 1 fit | 3 epochs | 5 epochs | 20% of all 3,420 development rows |
| Final refit | 1 fresh fit | Selected count | Selected count | None; all 3,420 rows used |

An **epoch** is one pass over that run's fitting rows. After each epoch, the
validation rows are evaluated without optimizer updates.
During **warmup**, backbone weights are frozen and the replacement classifier
head is trained at learning rate 0.001. During **full fine-tuning**, all network
weights become trainable, at learning rate 0.0001. Frozen weights do not imply
that every BatchNorm running statistic is frozen while the model is in train mode.

For the final fit: **2,736 fitting rows + 684 validation rows**. Rows are split
after augmentation. The best validation macro-F1 checkpoint is retained across
warmup and fine-tuning; the best warmup checkpoint is reloaded before fine-tuning.
An improvement replaces the checkpoint. Training still runs the configured
epochs; this is checkpoint selection, not early stopping.

**Final refit:** initialize a fresh pretrained model and train on all 3,420 rows
using the macro-F1-selected warmup/fine-tuning epoch counts. Test scores never
choose that schedule. A warmup-only winner produces a warmup-only refit.
Weighted-F1 saves a secondary reference checkpoint, not the final schedule.
See [DL Training Protocol](dl_training_protocol.md) for checkpoint and refit files.

**Outputs:** `01_Python/outputs_resnet18_thesis/`, with:

- `kfold_train_only/`: fold checkpoints, `outer_cv_scores.csv`, outer predictions,
  fold confusion matrices and learning curves.
- `final_train/bird_species_resnet18_FINAL_best.pth`: the fresh full-pool refit model.
- `final_train/FINAL_SELECTION_loss_curve.png` and `FINAL_SELECTION_acc_curve.png`: epoch histories.
- `final_train/split_membership.csv`, pool/protocol records and `model_analysis.json`:
  reproducibility and architecture/training metadata.
- `test_eval/`: test metrics, confusion matrix, report and prediction audit.

### 5. Train DL: Partial Fine-Tuning

**Script:** `dl_transfer_learning_bird_classifier_partial_finetune.py`

Repeat stage 4 with a separate ResNet18: five CV fits (2+3 epochs each), followed
by selection (3+5 epochs) and a fresh full-pool refit. Same pool, validation fraction and optimizer
learning rates. Fine-tuning updates the classifier head plus the last **four
parameter-bearing internal modules**, not four entire ResNet blocks.

**Outputs:** the same structure under
`01_Python/outputs_resnet18_thesis_partial_finetune/`.

Across both DL configurations there are **14 model fits**: ten CV fits, two
selection fits and two full-pool refits. Selection/CV total **66 epochs**;
refits add their selected epoch counts (1 to 8 each). CV uses a shorter budget
and evaluates inner-selected checkpoints without an outer-development refit,
so its scores describe that shorter procedure, not the final refit procedure.

### 6. Select SL Features and Export 29 Variants

**Script:** `sl_feature_comparison.py`

Selection/ranking uses the **1,140 original balanced training images only**.
Augmented and test rows do not determine those rankings. The selected columns
are then applied to original, augmented and test rows for export.

| Selection family | Exported variants | Count |
| --- | --- | ---: |
| ANOVA, Fisher, MI, PCA-based ranking | Top 100, 80 and 50 for each filter | 12 |
| Filter consensus | Top 100, 80 and 50 | 3 |
| SVM and k-NN wrappers | RFECV selection plus top 100, 80 and 50 for each | 8 |
| Random Forest and XGBoost importance | Top 100, 80 and 50 for each | 6 |
| Total, with XGBoost available | Feature-column variants, not new image collections | 29 |

PCA ranks original descriptors using loadings from three components; exported
columns remain original features, not PCA components. Wrapper selection performs
an initial five-fold grid search, five-fold RFECV, then a five-fold search on the
selected subset. RFECV removes features in steps of 10, down to a minimum of 5;
its total work depends on the feature count and elimination path.

**Outputs:** `sl_results/filters/`, `wrappers/`, `embedded/`,
`stage_comparison/`, `diagnostics/`, `exploratory/` and
`training_ready_datasets/training_ready_dataset_manifest.csv`, alongside the
selected CSVs and plots. Each current exported CSV is expected to retain 3,569
rows: 3,420 development rows + 149 test rows. Feature-list lengths can be smaller
than a requested top-k if fewer eligible features are available.

### 7. Train SL on Every Exported Variant

**Command:** `sl_training_pipeline.py --run-exported-batch`

For each variant, split the 3,420 development rows into **2,736 fitting rows
(80%) and 684 internal-validation rows (20%)**. Augmented and original rows can
appear on either side. Run five-fold GridSearchCV on the fitting part, optimizing
macro-F1. SVM and k-NN learn scaling inside their pipelines; tree models do not
use StandardScaler.

| Model | Hyperparameter combinations | Search fits per variant (5 folds) |
| --- | ---: | ---: |
| SVM RBF | 16 | 80 |
| k-NN | 16 | 80 |
| Random Forest | 24 | 120 |
| XGBoost | 8 | 40 |
| Total | 64 | 320 |

For 29 variants this is **9,280 candidate-fold fits**, plus best-model fitting,
probability calibration where applicable, and final refits. Feature-selection
work and independent nested CV are additional. This explains why "116 models"
does not mean only 116 training operations.

Evaluate tuned models on the 684 validation rows; choose by validation macro-F1,
not test performance. Refit each model with its chosen settings on **all 3,420
development rows**, then evaluate on the separate test set. With four available
families, 29 x 4 = **116 final saved SL models**. SL has no neural-network warmup
or epochs; Random Forest tree counts and XGBoost boosting rounds are different
training controls.

**Outputs:** `sl_outputs_batch/batch_TIMESTAMP/` contains batch comparison CSVs,
`selected_model.json` and `per_dataset/` results. Each variant has its validation
and test comparisons, `models/*_train_only.pkl`, `models/*_final.pkl`, metadata,
prediction audits, confusion matrices, grid-search results and timings.
`sl_outputs_batch/latest_batch_run.txt` points to the latest batch.

### 8. Independently Validate SL With Nested CV

**Command:** `thesis_validation.py sl-cv --method mi --features 50 ...`

This is a **separate MI-top-50 experiment**, not independent CV for every one of
the 29 exported variants. It repeats feature selection inside the search pipeline.

- Outer CV: five folds of the 1,140 originals; each fold holds out 228 originals.
- Outer development: 912 originals + their 1,824 views = 2,736 fitting candidates.
- Inner CV: five folds choose hyperparameters, fitting MI selection anew inside
  each candidate/fold pipeline. Validation uses held-out originals only;
  their augmented views are excluded from fitting.
- Refit the selected pipeline on the outer development data, then score the
  228 outer held-out originals. Repeat for each of the four model families.

This produces **20 outer model scores** (four families x five folds), from
**1,600 inner candidate-fold fits** (320 per outer fold x five), plus 20 refits.
It is one five-fold partitioning, not repeated CV over many random seeds.

**Outputs:** `sl_nested_cv/outer_cv_scores.csv`, selected hyperparameter JSONs,
fold confusion matrices, protocol/pool metadata and an outer-score plot. These
evaluation fits are not the final deployment `.pkl` files from stage 7.

### 9-10. Compare Models Within Each Family

**Script:** `thesis_validation.py compare ...`, called twice.

- SL: four MI-top-50 model families, six pairwise comparisons.
- DL: full versus partial fine-tuning, one pairwise comparison.

Use five paired **outer-fold macro-F1 scores** per model. The current test is a
Nadeau-Bengio corrected paired t-test with Holm multiple-comparison adjustment,
not ANOVA or Wilcoxon. Five folds provide limited statistical power. There is
no combined statistical SL-versus-DL ranking/test in this stage.

**Outputs:** `sl_statistics/` and `dl_statistics/`, each containing
`outer_scores.csv`, `family_cv_summary.csv`, `within_family_tests.csv`,
`statistical_protocol.json` and `outer_cv_scores.png`.

### 11. Predict the Test Images and Analyze Shared Errors

**Script:** `compare_dl_sl_models.py`

Load the two final DL checkpoints and all four SL models from the feature-variant
run containing the batch's validation-selected winner. This is **six evaluated
models**, not all 116 SL models and not necessarily each SL family's separately
best feature variant. No models are retrained here.

The runner passes `--balance-test`: final metrics and training-script summaries
use the **same 132 untouched test images**, 11 per species. Full prediction audits
retain all 149 images. Identity manifests prove matching, rather than equal counts
alone. See [Matched Test Pool](shared_test_pool.md); `--no-balance-test` explicitly
requests all-image metrics when running the comparison script directly.

**Outputs:** `01_Python/family_evaluation_results/`:

- `sl/` and `dl/`: family-specific summaries, confusion matrices and predictions.
- `image_error_analysis/all_predictions.csv`: every evaluated model/image result.
- `image_error_matrix.csv`: correctness by image and model.
- `image_error_frequency.csv`: number/group of failures per image.
- `species_error_summary.csv`, `confusion_pairs.csv`: species-level patterns.
- `annotation_template.csv`: manual condition labels, such as occlusion or blur.
- `error_gallery.html`: visual inspection of the failing images.

Image keys connect SL and DL predictions to the same species/filename. This is
descriptive shared-error analysis, not a claim of statistical superiority across
families. Condition explanations require actual annotations/inspection; filenames
alone do not establish why an image failed. The gallery uses ordinary predictions,
without rejection. Training may also write rejection-policy artifacts, but these
do not redefine correctness for this shared-image report.

## The Three Meanings of Validation

| Procedure | Purpose | Are held-out source images separated from augmented fitting views? |
| --- | --- | --- |
| Ordinary SL 20% / DL 20% validation | Select settings/checkpoints | No: rows are split after pooling augmentation |
| SL grid CV inside batch training | Tune settings for each exported feature list | No: ordinary mixed-row folds |
| Independent outer CV | Assess a training procedure without using final test | Yes, for the recorded original-image identities |

DL outer CV uses 912 originals plus their views for development, splits that
2,736-row pool into **2,188 fitting + 548 internal-validation rows**, trains for
2+3 epochs, then evaluates the selected checkpoint on the 228 outer originals.
The inner checkpoint split can mix related views; the outer holdout cannot.

Ordinary mixed-row validation can therefore be optimistic. Main SL feature
rankings also precede the batch's internal split, so those internal scores are
not fully independent feature-selection evidence. The separate nested MI-top-50
experiment addresses that issue for its own protocol, not all 29 variants.

Source isolation is only as good as the recorded identity: if multiple crops of
one original photograph have different filename keys, this alone does not group
them as one photograph. Test data must remain excluded from model/feature choice;
repeatedly changing the experiment to improve test scores would invalidate that
role even though the scripts never fit on test rows.

## Results, Logs and Reproducibility

Seed 42 controls the shared selection and configured random procedures; DL folds
also use fold-specific seeds. Fixed augmentation provenance makes the saved image
views reproducible. A fixed seed is not a guarantee of bit-identical results
across hardware, libraries or GPU kernels.

The runner saves commands, status and elapsed times under
`01_Python/pipeline_runs/run_TIMESTAMP/`. Exact runtime depends on hardware and
data; the epoch/fit counts above are not minute estimates. SL batch runs get new
timestamped directories. Several feature, DL and report outputs have fixed paths
and can be overwritten by a rebuild. Preserve a snapshot before rerunning when
historical outputs must remain available.

## Code Anchors for Future Corrections

- [Runner and execution order](../run_pipeline.py): `build_stages`.
- [Shared original pool](../shared_training_pool.py).
- [Augmentation and CV handling](../shared_augmentation.py).
- [Full DL settings](../dl_transfer_learning_bird_classifier.py): `main`, final training, `run_kfold_on_train`.
- [Partial DL settings](../dl_transfer_learning_bird_classifier_partial_finetune.py): `main`, `fine_tune_last_internal_layers`.
- [Feature selection/export](../sl_feature_comparison.py): `export_training_ready_feature_datasets`.
- [SL tuning and batch](../sl_training_pipeline.py): `make_model_spaces`, `run_training_pipeline`.
- [Independent CV/statistics](../thesis_validation.py): `run_sl_nested_cv`.
- [Final test evaluation](../compare_dl_sl_models.py): `main`.
- [Shared-image reports](../image_error_analysis.py): `summarize_image_errors`.
