# `dl_transfer_learning_bird_classifier.py`

This is the main DL training experiment. It reads an ImageFolder dataset with `train/<species>` and `test/<species>` folders, creates an internal validation split from training data, and trains a pretrained ResNet18 bird-species classifier.

The workflow supports reproducible splits, optional balancing, head warm-up followed by fine-tuning, optional stratified K-fold reporting, learning-curve plots, a saved checkpoint, and final test metrics including a confusion matrix and classification report.

Update the paths and hyperparameters in `main()` before execution. Its output checkpoint is the artifact consumed by DL prediction helpers.
