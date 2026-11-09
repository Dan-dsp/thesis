# `dl_transfer_learning_bird_classifier_partial_finetune.py`

> Current default, 2026-11-05: [five mixed-row folds, 3+5 epochs each, mean-epoch
> refit](dl_cv_workflow.md). Partial fine-tuning still uses the last four internal
> parameter-bearing modules plus the head. The older outer-CV/holdout description
> below is historical and is no longer the `main()` workflow.

Training now uses originals plus the exact saved image views also used by SL.
Extra online random flips/rotations/color jitter are disabled. The whole mixed
pool is split at row level for ordinary fitting/validation: 2,736 / 684 rows.
Related versions may cross these sides, so validation can be optimistic. Separate
outer CV still reserves originals and excludes their views from development.
No validation assignment is precreated. Final training resets its
seed independently of CV; cuDNN deterministic mode is enabled. Old checkpoints
must be retrained for this augmentation protocol.
See [Shared Augmentation](shared_augmentation.md).

Train undersampling now reuses the same persistent image list as full DL and SL,
without changing the original sampling/RNG order. Final/CV folders save the pool
and final output saves internal split membership. See [Shared Training Pool](shared_training_pool.md).

Existing loss/accuracy curves and test confusion matrices remain. New independent
outer-fold confusion matrices and `outer_cv_scores.png` are saved in
`kfold_train_only`. Saved plots use the non-GUI Agg backend.
See [Result Plots](result_plots.md).

New runs save parameter counts and partial-layer settings, select by macro-F1 and refit a fresh model on all 3,420 training rows. Weighted-F1 checkpoints are secondary; rejection is disabled without independent calibration. See [DL Training Protocol](dl_training_protocol.md). Test audits cover every image. K-fold checkpoint selection uses inner validation, followed by independent outer-fold evaluation. See [Thesis Evaluation Protocol](thesis_evaluation_protocol.md).

This is an alternative DL training experiment based on the same ResNet18/ImageFolder approach as `dl_transfer_learning_bird_classifier.py`. Its distinguishing choice is partial fine-tuning: it controls which backbone layers are trainable rather than always fine-tuning the whole network.

It performs internal validation, optional K-fold reporting, final training, and test evaluation, producing checkpoints and plots. Use it to compare a more constrained fine-tuning strategy against the main training script; it is not required by the prediction pipeline.
