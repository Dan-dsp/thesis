# `sl_models_evaluation.py`

Older standalone helper for evaluating saved shallow-learning models. It loads a feature CSV and Joblib models, evaluates the test rows, produces metrics and confusion matrices, saves a comparison table, and can create PCA visualizations.

The current `sl_training_pipeline.py` already trains and evaluates models as part of its run, so this script is optional for extra analysis rather than a required pipeline stage.
