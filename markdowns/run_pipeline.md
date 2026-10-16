# `run_pipeline.py`

Sequential launcher for existing scripts. Run it from any working directory, or
open this file in VS Code and click Run Python File. The launcher uses the same
Python interpreter as itself and sets every child's working directory to
`01_Python`. It never starts training stages in parallel.

## Recommended Current Run

From `F:/01_Univalle/01_TG`:

```powershell
python 01_Python/run_pipeline.py
```

Default sequence:
1. `sl_feature_comparison.py`: train-only feature selection, including Fourier features.
2. `sl_training_pipeline.py --run-exported-batch`: train/test every exported feature variant.
3. `compare_dl_sl_models.py`: reuse saved DL checkpoints, evaluate both families independently,
   and generate the shared-image error gallery and tables.

Extraction is skipped because the saved full feature table already contains the
Fourier descriptors. DL retraining is skipped because the SL name-filtering fix
does not affect DL. The default explicitly uses your historical bbox full-DL
checkpoint in `outputs_resnet18_thesis_dl_bbox`, and the saved partial-fine-tuning
checkpoint. It prints the paths before running; it never picks the no-bbox model.

## Other Modes

```powershell
python 01_Python/run_pipeline.py --dry-run
python 01_Python/run_pipeline.py --full
python 01_Python/run_pipeline.py --reports-only
```

`--dry-run` only prints the plan; no stages run and no files are written.

`--full` adds handcrafted-feature extraction and both DL training scripts before
SL selection/training and reporting. DL includes CV with the trainers' current
RUN_KFOLD settings. The final comparison uses the newly generated checkpoints,
not the historical full-DL checkpoint. This mode can take many hours and overwrites
the trainers' fixed DL output files. Existing SL batches remain timestamped.

`--reports-only` reuses saved SL/DL models and performs inference/reporting only.
It does not correct outdated SL feature selection or retrain old models.

Use `--dl-full-model path/to/model.pth` and `--dl-partial-model path/to/model.pth`
to override saved checkpoints in default or report-only mode. Relative paths are
resolved from `01_Python`. Overrides are refused with `--full`, to avoid comparing
old checkpoints after retraining new ones.

## Statistical Analysis

```powershell
python 01_Python/run_pipeline.py --full --with-statistics
```

Adds nested SL CV with MI top-50, SL within-family statistics, and DL within-family
statistics. The nested SL protocol is not an independent test of all 29 exported
variants. No joint SL/DL significance test is run. In default mode, this option
requires existing independent DL outer-CV scores; otherwise use the full mode.
It cannot be combined with `--reports-only`, since nested CV trains models.

## Failure Handling and Logs

Input scripts and reused DL checkpoints are checked before long-running stages.
Every stage must finish successfully before the next starts. On failure or a
keyboard interruption, later stages do not run. Inspect:

`01_Python/pipeline_runs/run_TIMESTAMP/`

Each stage has a UTF-8 terminal log, and `status.json` records the exact commands,
completion/failure status and elapsed time. The launcher does not automatically
resume failed runs or skip earlier completed stages on restart.

## Local HTML Report

After successful evaluation, open:

`01_Python/family_evaluation_results/image_error_analysis/error_gallery.html`

This is a local HTML file, not a hosted website. It only exists after reporting
finishes. Double-click it or open it in your browser. It reads images from local
disk, so keep their paths available. See [Image Error Analysis](image_error_analysis.md).

No mode splits the dataset, changes train/test assignments or reruns the bird
detector/cropping stage. All modes assume `dataset_bbox/train` and `dataset_bbox/test`
are prepared. Rejection behavior is not modified by this launcher.
