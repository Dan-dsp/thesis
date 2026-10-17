# Thesis Evaluation Protocol

SL and DL are studied independently. A descriptive statement about higher or lower observed
performance across families is allowed, accompanied by dataset, model-size and compute context.
Do not create a pooled ranking or cross-family significance test.

## Selected Features in SL

sl_main.py runs extraction, train-only feature selection and manifest-driven batch training.
sl_training_pipeline.py defaults to consensus top-50; the original full-feature CSV is refused
as a direct training input. Regenerate selection before training:

```powershell
python sl_feature_comparison.py
python sl_training_pipeline.py --run-exported-batch
```

New manifests record selection_split=train, selection_protocol=train_only_v4_shared_augmentation
and training_pool_id. SL selection, tuning, final refit and nested CV all use the
same DL-ordered balanced development pool (1,140 images, 95/species). The persistent
list is training_protocol/balanced_training_pool_seed_42.csv with JSON provenance.
Regenerate older exports and results. The shared augmentation migration also
requires descriptor extraction and both families' retraining; use run_pipeline.py.
No validation membership is preassigned. See [Shared Training Pool](shared_training_pool.md)
and [Shared Augmentation](shared_augmentation.md).
Legacy names match only f followed by digits, so fourier_1 through fourier_20
remain descriptive feature candidates. Earlier v1 exports excluded them and are
refused by batch training. Re-run selection and training, not extraction, when
the raw feature CSV already contains these Fourier columns. Re-run nested SL CV
and downstream statistics/error reports if they were previously generated.
Older manifests fail validation and must be regenerated. This version marker documents
provenance; it is not a cryptographic guarantee. The usual export has up to 29 variants
when XGBoost is available. Internal validation used by the batch has already influenced
feature selection; use nested evaluation below for statistical evidence.

## Separate Model Analyses

Each SL run writes MODEL_model_analysis.json: selected feature count, estimator settings
and model-specific complexity (trees/nodes, support vectors, or stored samples). A neural
network parameter count is not assigned to k-NN, SVM or forests.

Each new DL final checkpoint stores training_analysis and writes model_analysis.json:
ResNet18 architecture, total parameters, trainable parameters in the final training stage,
Adam/cross-entropy procedure, rates, epochs, batch size, device and balancing settings.
New model analyses also record dataset sizes, saved model bytes and refit/training duration.
Existing timing files continue to report SL costs. The DL epoch console includes durations;
a comparable GPU FLOP or measured peak-memory analysis is not inferred from parameter counts.

compare_dl_sl_models.py is retained as a compatibility filename. Its new outputs are
family_evaluation_results/sl and family_evaluation_results/dl, with no joint ranking.
Its SL default resolves the validation-selected run from the latest batch.
The full-DL checkpoint path matches outputs_resnet18_thesis/final_train, as used
by the full-fine-tuning trainer; partial fine-tuning retains its separate folder.
The two evaluation populations and training settings must be stated when mentioning
descriptive performance across families. Existing comparison_results are historical.

## Cross-Validation and Statistics

The existing SL grid-search CSVs are tuning diagnostics. Their winning folds must not be
used as independent statistical evidence, because they participated in selecting settings.
For a thesis evaluation, run feature selection inside inner cross-validation:

```powershell
python thesis_validation.py sl-cv --method mi --features 50 --out-dir F:/01_Univalle/01_TG/sl_nested_cv
python thesis_validation.py compare F:/01_Univalle/01_TG/sl_nested_cv/outer_cv_scores.csv --out-dir F:/01_Univalle/01_TG/sl_statistics
```

This optional, expensive procedure uses only original train rows, fits SelectKBest inside
each inner training fold, tunes each of the four classifiers, and evaluates macro-F1 on
the outer fold once. It currently supports MI and ANOVA subsets; it does not silently
claim independent evaluation of all 29 preselected variants. Its development
population is the shared balanced pool, not all 1,235 original train rows.

Both DL training scripts now split each outer training fold internally for checkpoint/epoch
selection. The held-out outer fold is evaluated after loading the best inner-validation
checkpoint. Enable RUN_KFOLD in each script to create new outer_cv_scores.csv files.
The historical kfold_summary best_val_acc values are checkpoint-selection scores and cannot
replace outer scores. Use the same original images, balancing, folds and seed for DL variants.

Pass both new DL outer_cv_scores.csv files to thesis_validation.py compare with --out-dir.
The script refuses mixed families, duplicate records, unmatched partition hashes and scores
not marked independent_outer. Scores must all represent macro-F1.

The test is a Nadeau-Bengio corrected paired t-test, with Holm correction across model pairs.
The correction accounts approximately for overlapping training data in k-fold CV. A plain
Wilcoxon or ANOVA on correlated fold scores is not automatically appropriate. Five folds
offer limited power; non-significance does not establish equivalence. Degenerate nonzero
differences with zero variance are marked untestable. Choose the protocol in advance.

Reference: [scikit-learn statistical comparison](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html).

## Rejection

Confidence is the uncalibrated probability of the predicted class (softmax for DL). SVM
predict and probability argmax can disagree; the SL audit scores the actual predicted label.
The default policy selects the greatest validation coverage meeting 90% empirical accepted
accuracy and a minimum support of max(10, 10% of validation samples), capped at validation size.
If no threshold qualifies, the policy explicitly records failure and rejects all predictions.
This is an empirical validation target, not a guaranteed test accuracy or out-of-distribution detector.

SL thresholds use the train-only model's internal validation before the final full-train refit.
Refitting can change confidence; evaluate resulting coverage drift without retuning on test.
DL thresholds use the selected final checkpoint on its internal validation. The validation
data also selected the checkpoint, so confidence calibration claims require another held-out set.
SL selected_model.json selects by internal-validation macro-F1, including the best dataset
variant at batch level. DL candidates all support rejection; identify the best DL configuration
using new DL outer-CV summaries rather than test accuracy.

predict_sl_with_rejection.py accepts a selected_model.json and a feature CSV and writes
confidence, rejected and decision columns. predict_single_image.py uses checkpoint policy
by default; use_rejection=False disables it. Legacy DL checkpoints have no learned policy
and remain without rejection until retrained or calibrated on held-out training data.

## Error Analysis

SL and DL write predictions/MODEL_test_predictions.csv for every test image, even when
the conventional performance table uses a balanced test subset. Raw mistakes remain in
MODEL_test_errors.csv regardless of rejection. MODEL_rejection.json reports coverage,
accepted accuracy, raw accuracy and the fraction of all images correctly classified and accepted.

```powershell
python classification_analysis.py path/to/sl_test_predictions.csv path/to/dl_test_predictions.csv --out-dir error_review
```

The output includes descriptive_error_overlap.csv, image_error_frequency.csv, error_review.csv
and annotation_template.csv. Cross-family overlap is descriptive error inspection only.
sample_key is species/original_filename. Match only identical original images; if crops have
different names, supply a verified correspondence instead of assuming identical identity.

The current analysis is image-focused: individual-bird identity is not required.
Fill annotation_template.csv with semicolon-separated conditions (blur;occlusion;
backlighting;small_bird, for example) and notes, then rerun with --annotations path/to/filled.csv.
condition_summary.csv reports per-model condition-specific errors and denominators.
Annotate successful images too. Observed condition associations do not prove causation.

compare_dl_sl_models.py now exports full-test raw predictions and automatically creates
image_error_analysis/error_gallery.html, image_errors.png, shared/complementary error lists,
species confusion summaries and explicit unmatched-image reporting. The analysis ignores
rejection decisions and uses raw species predictions; rejection helpers are otherwise unchanged.
See [Image Error Analysis](image_error_analysis.md). Legacy bird_id annotations are still
readable for compatibility, but the new template does not request them.

New code does not regenerate historical experiments automatically. Re-run selection and
training to obtain the new artifacts. Changing a model after inspecting test errors requires
a new untouched evaluation set for the next confirmatory claim.
