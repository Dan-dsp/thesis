# `compare_dl_sl_models.py`

Evaluates trained deep-learning and shallow-learning models on a comparable test set and writes a combined report. It is not called by either training pipeline; run it after both pipelines have produced their model artifacts.

It loads ResNet18 checkpoints for DL and Joblib model runs plus their feature metadata for SL. The script can balance samples per class, calculates accuracy, macro precision/recall/F1 and confusion matrices, then saves tables and plots under its configured output directory.

Use the command-line options in `--help` to point it at the DL checkpoint, SL run directory, and dataset. It belongs to cross-pipeline evaluation, not dataset creation or training.
