# DL Selection, Refit and Matched Testing

> Superseded default (2026-11-05): see [the current five-fold workflow](dl_cv_workflow.md).
> Both trainers now call `dl_cv_workflow.py`. `fit_with_f1_checkpoints` uses
> `phase_specific=True` there, preserving separate warmup/fine-tuning winners;
> `refit_full_pool` receives the rounded arithmetic-mean schedule from five folds.
> The description below documents the historical holdout/outer-CV workflow only.

Both DL trainers now use this protocol (`dl_training_protocol.py`).

1. Reuse the shared balanced training pool: 1,140 originals and 2,280 saved augmented views. Test images never enter this pool.
2. Stratify the pooled rows into 80% fitting / 20% internal validation: 2,736 / 684 rows. Related views can cross this split; internal scores can therefore be optimistic. This preserves the requested pooled-row protocol.
3. Run the configured selection budget (3 warmup + 5 fine-tuning epochs). Compute macro-F1 and weighted-F1 over the entire validation epoch. Macro-F1 selects the primary checkpoint; weighted-F1 saves a separate reference checkpoint. Accuracy remains descriptive only. Ties retain the earliest checkpoint.
4. Reload the best macro-F1 warmup checkpoint before fine-tuning. Record the warmup and fine-tuning epoch counts that produced the winning macro-F1 checkpoint. A warmup-only winner implies zero fine-tuning epochs for refit.
5. Initialize a fresh pretrained ResNet18 and train on ALL 3,420 development rows using that fixed schedule. Do not reuse the selection model weights, consult validation/test scores, or select another checkpoint. Partial training retains its restricted layers; full training unfreezes the full backbone.
6. Evaluate the resulting full-pool model on the same 132 test identities used by SL (11 per species). Keep full 149-image prediction audits for error inspection.

`f1_score` is a function name, not an additional averaging rule. SL uses `average="macro"`; this primary criterion now matches DL. Weighted F1 is explicitly labeled and is not used to select the final refit schedule.

## Outputs

Under each DL output root:

- `final_train/selection/best_macro_f1.pth`: internal-validation winner.
- `final_train/selection/best_macro_f1_weighted_f1.pth`: secondary weighted-F1 winner.
- `final_train/selection/best_macro_f1.history.csv` and `.f1.png`: selection histories.
- `final_train/FINAL_SELECTION_loss_curve.png` and `FINAL_SELECTION_acc_curve.png`: selection curves.
- `final_train/bird_species_resnet18_FINAL_best.pth`: fresh full-pool refit, retaining the existing prediction-compatible filename.
- `final_train/refit_history.csv`, `refit_protocol.json`, `refit_membership.csv`: training history, selected schedule and exact full-pool identities.
- `final_train/model_analysis.json`: parameters, hyperparameters and timing.
- `test_eval/balanced_test_pool.csv` and `.json`: identities, counts and cohort hash.

The final refit has no independent validation set. Its stored selection scores are explicitly selection-stage scores, not validation of the refitted weights. Rejection is disabled for the refit: no old-model confidence threshold is silently reused.

## Cross-Validation

Five outer folds still hold out original source identities and their augmented versions. Inner checkpoint selection now uses 20% validation and macro-F1, with the existing 2 warmup + 3 fine-tuning budget. These outer scores describe the shorter inner-selected procedure; outer folds do not perform the final full-pool refit.

Across both configurations: ten CV models, two selection models and two fresh refit models. Selection/CV consume 66 aggregate epochs; the two refits add the validation-selected epoch counts (1 to 8 each). This is not one model trained for that aggregate duration.

## Regenerate Results

From `01_Python`:

```powershell
python dl_transfer_learning_bird_classifier.py
python dl_transfer_learning_bird_classifier_partial_finetune.py
python thesis_validation.py compare outputs_resnet18_thesis/kfold_train_only/outer_cv_scores.csv outputs_resnet18_thesis_partial_finetune/kfold_train_only/outer_cv_scores.csv --out-dir ../dl_statistics
python run_pipeline.py --reports-only
```

This regenerates DL models/CV/test results, DL within-family statistics, then the reports. Existing SL models, descriptors, feature selection and saved augmentation do not need rebuilding for these changes. Old DL results remain historical until the trainers finish. `--reports-only` alone does not refresh CV statistics.

See [Matched Test Pool](shared_test_pool.md).
