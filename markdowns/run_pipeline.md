# run_pipeline.py

The default is now a complete SL/DL experiment rebuild, not SL-only refresh with
old DL checkpoints. Run from `F:/01_Univalle/01_TG`:

```powershell
python 01_Python/run_pipeline.py
```

Opening `run_pipeline.py` and clicking Run Python File does the same thing.
The existing filename is `run_pipeline.py`, not `run_pipelin.py`.

## Default Sequence

1. Verify the shared balanced original-image pool (`shared_training_pool.py`).
2. Generate two reproducible shared image views (`shared_augmentation.py`).
3. Re-extract original and augmented descriptors (`sl_dataframe_main.py`).
4. Train/CV/evaluate full-fine-tuning DL.
5. Train/CV/evaluate partial-fine-tuning DL.
6. Select SL features on original balanced train rows and export reduced CSVs.
7. Train/test the exported SL batch with fitting-only augmented descriptors.
8. Run independent nested SL CV, MI top-50, for the four classifiers available.
9. Analyze SL outer-fold scores within the SL family.
10. Analyze DL outer-fold scores within the DL family.
11. Evaluate saved models and generate shared-image error tables/HTML gallery.

Nested SL MI top-50 CV is not independent CV for every one of the 29 variants.
No combined SL/DL significance test is performed. Final reporting uses newly
trained DL checkpoints in `outputs_resnet18_thesis` and
`outputs_resnet18_thesis_partial_finetune`, and the newly generated SL batch.

## Options

```powershell
python 01_Python/run_pipeline.py --dry-run
python 01_Python/run_pipeline.py --skip-statistics
python 01_Python/run_pipeline.py --reports-only
```

- `--dry-run`: print commands without checks, training or file writes.
- `--skip-statistics`: omit nested SL CV and both statistical-analysis commands;
  the DL trainers still run their enabled CV stages.
- `--full` and `--with-statistics`: retained explicit aliases for default behavior.
- `--reports-only`: inference/reporting on existing artifacts only. Its full-DL
  default is the rebuilt checkpoint under `outputs_resnet18_thesis`. Historical
  checkpoints need an explicit `--dl-full-model path/to/model.pth` override.
  Checkpoint overrides are permitted
  only in reports-only mode. Old reports do not establish the new augmentation
  protocol was used.

## Runtime And Outputs

This is a long training run, potentially many hours. It overwrites fixed DL
output files and extracted feature CSVs. SL batches retain timestamped folders.
Augmented samples increase fitting costs; selection rankings still use originals.

Every stage runs sequentially with the launcher's Python interpreter and
`01_Python` working directory. `pipeline_runs/run_TIMESTAMP/` stores terminal
logs and `status.json` with commands, durations and completion/failure status.
Failures stop subsequent stages. Restarting does not automatically resume.

The HTML report is:
`01_Python/family_evaluation_results/image_error_analysis/error_gallery.html`.

No detector, crop, rename or original train/test splitting stages are rerun.
Prepared `dataset_bbox/train` and `dataset_bbox/test` remain prerequisites.
No validation-assignment file is precreated. Rejection behavior is unchanged.
See [Shared Augmentation](shared_augmentation.md) for sampling, seed controls,
fold isolation and remaining methodological limitations.
