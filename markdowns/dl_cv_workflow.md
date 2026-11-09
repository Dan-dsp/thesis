# DL workflow: five-fold epoch selection and fresh refit (2026-11-05)

This is the default for both DL training scripts. It replaces the previous
independent outer-CV plus separate single-holdout selection sequence. Historical
helpers remain explicitly available in the trainer modules, but `main()` does
not call them. `main.tex` is not automatically rewritten by the runner.

## One command

```powershell
cd F:\01_Univalle\01_TG\01_Python
python rerun_dl_workflow.py
```

Preview without training/writing: `python rerun_dl_workflow.py --dry-run`.
The runner validates and reuses the existing saved augmentation and SL models.
It checks that their training-pool and augmentation identifiers match. It does
not rebuild descriptors, feature selection, SL models, or the image dataset.

The five sequential stages are full DL, partial DL, descriptive selection-score
comparison, balanced-test SL/DL comparison, and full-test SL/DL comparison.
A failing stage stops the workflow. Each stage has a streamed log and status.

## Exact training procedure

1. Build the same saved mixed development pool of originals and augmented views.
   For the current dataset this is 1,140 + 2,280 = 3,420 rows.
2. StratifiedKFold creates five row-level folds with seed 42. Original/augmented
   relatives are deliberately NOT grouped. Each fold uses 2,736 fitting rows and
   684 validation rows for the current dataset. Fold membership stays fixed over
   all epochs; only fitting-row order is shuffled between epochs.
3. Initialize a fresh ImageNet-pretrained ResNet18 and new classification head
   for each fold (seed 42 + fold number).
4. Run exactly three warmup epochs. Validate after each epoch and retain the
   highest macro-F1 checkpoint; ties keep the earliest epoch.
5. Restore that fold's best warmup checkpoint. Run exactly five fine-tuning
   epochs. Select the best fine-tuning checkpoint independently of the warmup
   score. Full fine-tuning unfreezes the backbone; partial fine-tuning unfreezes
   the last four parameter-bearing internal modules plus the head.
6. There is no early stopping. All eight selection epochs run in every fold.
7. Compute the arithmetic mean of the five best warmup epoch numbers, and
   separately of the five best fine-tuning epoch numbers. Round each to the
   nearest integer (half-up). These are phase-local epoch numbers, not total
   elapsed epochs, and this is a mean, not a median or the best fold's schedule.
8. Initialize a fresh pretrained model (seed 42) and refit on every development
   row using the two fixed epoch counts. No fold weights, validation monitoring,
   or test data are used to select this refit. Save the final epoch's weights.
9. Evaluate the final model on the test set. Balanced reports use the existing
   identity-matched 132-image subset; full reports/audits cover all 149 images.

There are six trained model instances per variant: five fold models and one
final refit, not seven. The mean schedule is a practical rule, not a guarantee
of the optimum on the larger refit pool. Rejection remains disabled for fresh
full-pool refits because no independent calibration set is reserved.

## Results and provenance

Every full runner invocation uses a fresh directory:
`../dl_workflow_runs/run_<timestamp>/`. Old results are retained.

- `full/` and `partial/`: respective training outputs.
- `cv_selection/fold_N/best_warmup_macro_f1.pth`: best warmup state.
- `cv_selection/fold_N/best_finetune_macro_f1.pth`: best fine-tuning state.
- Each fold's `.history.csv` records phase, phase epoch, validation metrics,
  whether that epoch improved macro F1, and final selected-checkpoint flags.
- Each fold's `split_membership.csv`: every row's fitting/validation role.
- `cv_selection/fold_scores.csv`: both phase winners, metrics, epoch numbers,
  partition IDs, source paths, and shared-data identifiers.
- `cv_selection/cv_summary.csv`: mean and sample standard deviation per metric.
- `cv_selection/epoch_selection.json`: all five schedules, raw arithmetic means,
  rounding rule, final schedule and selection runtime.
- `final_train/bird_species_resnet18_FINAL_best.pth`: fresh deployable refit.
- `final_train/refit_protocol.json`, `refit_history.csv`, `refit_membership.csv`:
  schedule provenance, actual training epochs and full-pool membership.
- `dl_selection_comparison/`: paired descriptive fold results and plots.
- `comparison_balanced/` and `comparison_full_test/`: SL/DL metrics, confusion
  matrices, individual predictions and `image_error_analysis/error_gallery.html`.
- `workflow_manifest.json`: status, exact commands, SL run, source-code hashes,
  final checkpoint hashes and log location. `../dl_workflow_runs/latest_run.txt`
  points to the last successfully completed run.

To remake reports after a completed run without retraining:

```powershell
python rerun_dl_workflow.py --reports-only --run-dir F:/01_Univalle/01_TG/dl_workflow_runs/run_<timestamp>
```

Standalone trainers also support `--output-dir`; their defaults are absolute
project-root `outputs_resnet18_thesis` and `outputs_resnet18_thesis_partial_finetune`,
independent of the working directory. Runner comparisons pass exact checkpoint
paths and do not fall back to these defaults or older `01_Python/outputs_*`.

## Interpretation limits

These validation folds choose checkpoints. Related augmented views can cross
fold boundaries by the user's requested protocol. Scores can be optimistic and
are labelled `checkpoint_selection_not_independent`. They are NOT new outer-CV
scores. The runner does not generate corrected paired-test p-values or overwrite
historical `dl_statistics`/`outer_cv_scores.csv`. `thesis_validation.py compare`
continues to reject selection scores. Use final test reports for descriptive
comparison; do not tune the schedule using those test scores. The previously
inspected test-set history must remain disclosed in the thesis.
