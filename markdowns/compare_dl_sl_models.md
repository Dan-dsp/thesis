# `compare_dl_sl_models.py`

Every run now exports full-test raw per-image predictions under each family's
`predictions/` folder and automatically creates `image_error_analysis/`, including
an image gallery and shared-error matrix. Optional balanced metric tables remain
separate from these full-test audits. Rejection is not used for this report.
See [Image Error Analysis](image_error_analysis.md).

Use `--dl-full-model` or `--dl-partial-model` to select saved DL checkpoints
explicitly, including historical folders, without editing code or retraining.

Each family output folder also contains labeled confusion-matrix PNGs and
`summary_metrics.png`, without a pooled SL/DL plot. See [Result Plots](result_plots.md).

Evaluates saved classifiers into independent SL and DL reports. The compatibility filename remains, but the script no longer produces a joint ranking or a cross-family significance test.

It loads ResNet18 checkpoints and reduced-feature SL models. SL defaults to the validation-selected run from the latest batch. Outputs are family_evaluation_results/sl and family_evaluation_results/dl. Metrics across families may only be described with their dataset and training context.

Use --sl-run-dir to choose an explicit SL run, or --skip-sl / --skip-dl to evaluate one family. DL checkpoint paths remain configured in DL_MODELS. See [Thesis Evaluation Protocol](thesis_evaluation_protocol.md) for image errors, rejection and independent CV tests.

The full-fine-tuning checkpoint is loaded from `outputs_resnet18_thesis/final_train`,
matching the trainer's output folder. After regenerating Fourier-aware SL feature
subsets and retraining the batch, rerun this script; existing DL checkpoints can
be reused because the SL feature-name correction does not affect DL training.
