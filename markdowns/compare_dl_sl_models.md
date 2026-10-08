# `compare_dl_sl_models.py`

Evaluates saved classifiers into independent SL and DL reports. The compatibility filename remains, but the script no longer produces a joint ranking or a cross-family significance test.

It loads ResNet18 checkpoints and reduced-feature SL models. SL defaults to the validation-selected run from the latest batch. Outputs are family_evaluation_results/sl and family_evaluation_results/dl. Metrics across families may only be described with their dataset and training context.

Use --sl-run-dir to choose an explicit SL run, or --skip-sl / --skip-dl to evaluate one family. DL checkpoint paths remain configured in DL_MODELS. See [Thesis Evaluation Protocol](thesis_evaluation_protocol.md) for image errors, rejection and independent CV tests.
