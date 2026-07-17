# `sl_main.py` Guide

This file is the orchestrator for the shallow-learning workflow.

It lets you run the main stages from one Python process:

1. feature extraction
2. feature comparison / selection
3. model training

## Why this file exists

If you run each stage separately, Python has to start and import everything again each time.

Using [sl_main.py](f:/01_Univalle/01_TG/01_Python/sl_main.py) helps by:

- starting one process
- importing stages only when needed
- running the selected stages in sequence

## Current stage layout

### Feature extraction

- [sl_dataframe_main.py](f:/01_Univalle/01_TG/01_Python/sl_dataframe_main.py)

### Feature comparison / selection

- [sl_feature_comparison.py](f:/01_Univalle/01_TG/01_Python/sl_feature_comparison.py)
- [sl_feature_comparison_tools.py](f:/01_Univalle/01_TG/01_Python/sl_feature_comparison_tools.py)

Important detail:

- `sl_feature_comparison.py` is now the high-level workflow
- `sl_feature_comparison_tools.py` stores the lower-level methods

So the stage is still launched from `sl_feature_comparison.py`, but its implementation is now split for readability.

Important workflow update:

- this stage now also exports training-ready reduced CSV datasets
- it writes a manifest that can be consumed by the training pipeline batch mode

### Model training

- [sl_training_pipeline.py](f:/01_Univalle/01_TG/01_Python/sl_training_pipeline.py)

Important workflow update:

- this stage can still train all classifiers on one chosen CSV
- it can now also run all exported feature-set CSVs automatically in batch mode

## Structure

## `Stage`

A small dataclass that stores:

- the stage name
- the runner function

## `_run_stage(stage_name, runner)`

Purpose:

- print a start message
- run the stage
- measure elapsed time
- print a finish message

## `main(...)`

Arguments:

- `run_feature_extraction=True`
- `run_feature_analysis=True`
- `run_model_training=True`

Behavior:

1. Build the selected stage list.
2. Import only the enabled stage modules.
3. Run them in order.
4. Print total pipeline runtime.

Even though the argument name remains `run_feature_analysis`, the stage itself now corresponds to the broader feature-comparison and feature-selection workflow.

## Recommended usage

Use [sl_main.py](f:/01_Univalle/01_TG/01_Python/sl_main.py) when you want the full end-to-end process:

- feature extraction
- feature comparison / selection
- model training

The recommended order is now:

1. run feature extraction if the main feature CSV does not exist yet
2. run `sl_feature_comparison.py` to generate rankings and training-ready reduced CSVs
3. run `sl_training_pipeline.py` either:
   - on one selected CSV
   - or in batch mode across the full exported manifest

If you only need one stage, you can still run that stage file directly.

Practical examples:

- full comparison workflow:
  - `python sl_feature_comparison.py`
  - `python sl_training_pipeline.py --run-exported-batch`

- one chosen feature family:
  - `python sl_feature_comparison.py`
  - `python sl_training_pipeline.py --data-csv "F:/01_Univalle/01_TG/sl_results/training_ready_datasets/filters/consensus/consensus_top_100_training.csv"`

## Short Summary

[sl_main.py](f:/01_Univalle/01_TG/01_Python/sl_main.py) remains the top-level orchestrator.

Its main update is conceptual:

- the second stage is no longer just a small feature-analysis script
- it now launches the refactored multi-stage feature-comparison workflow split across
  - [sl_feature_comparison.py](f:/01_Univalle/01_TG/01_Python/sl_feature_comparison.py)
  - [sl_feature_comparison_tools.py](f:/01_Univalle/01_TG/01_Python/sl_feature_comparison_tools.py)
- the feature-comparison stage now prepares reduced training datasets for the training stage
- the training stage can now evaluate one selected feature set or all exported feature sets in batch mode
