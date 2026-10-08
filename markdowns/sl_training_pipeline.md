# `sl_training_pipeline.py`

Training now defaults to the reduced consensus top-50 CSV; batch mode requires a new train-only selection manifest. It writes per-image prediction/error audits, rejection coverage and accuracy, model-specific complexity, and a validation-selected model descriptor. See [Thesis Evaluation Protocol](thesis_evaluation_protocol.md).

Main SL model-training pipeline. It loads the handcrafted feature CSV or reduced feature-set CSVs, uses only the declared training rows to build an internal validation split, tunes classical models, and evaluates the selected models on the held-out test rows.

It supports SVM, Random Forest, k-NN, and XGBoost when installed. Each run records models, feature columns, metrics, confusion matrices, comparisons, and timing in a timestamped output directory. Batch mode consumes the feature-set manifest from `sl_feature_comparison.py`.

See `sl_training_pipeline_guide.md` for the detailed guide.
