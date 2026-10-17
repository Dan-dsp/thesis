# `dl_transfer_learning_bird_classifier.py`

Training now uses originals plus the exact saved image views also used by SL.
Extra online random flips/rotations/color jitter are disabled. Runtime fitting
subsets get their originals' views; validation, outer folds and test use only
originals. No validation assignment is precreated. Final training resets its
seed independently of CV; cuDNN deterministic mode is enabled. Old checkpoints
must be retrained for this augmentation protocol.
See [Shared Augmentation](shared_augmentation.md).

Train undersampling now creates/reuses the persistent shared image list also
consumed by SL. The original DL sample selection and RNG order are unchanged.
Final/CV folders save training_pool.csv/JSON; final output also saves actual
internal split membership. See [Shared Training Pool](shared_training_pool.md).

Existing loss/accuracy curves and test confusion matrices remain. New independent
outer-fold confusion matrices and `outer_cv_scores.png` are saved in
`kfold_train_only`. Saved plots use the non-GUI Agg backend.
See [Result Plots](result_plots.md).

New runs save architecture/parameter/hyperparameter analysis and validation-selected rejection in the final checkpoint. Test audits cover every image. K-fold runs now choose checkpoints on an inner split and report independent outer-fold macro-F1 in outer_cv_scores.csv. See [Thesis Evaluation Protocol](thesis_evaluation_protocol.md).

This is the main DL training experiment. It reads an ImageFolder dataset with `train/<species>` and `test/<species>` folders, creates an internal validation split from training data, and trains a pretrained ResNet18 bird-species classifier.

The workflow supports reproducible splits, optional balancing, head warm-up followed by fine-tuning, optional stratified K-fold reporting, learning-curve plots, a saved checkpoint, and final test metrics including a confusion matrix and classification report.

Update the paths and hyperparameters in `main()` before execution. Its output checkpoint is the artifact consumed by DL prediction helpers.
