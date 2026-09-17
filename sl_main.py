"""
sl_main.py

Orchestrator for the shallow-learning pipeline.

This file lets you run the whole pipeline in one Python process:
1) feature extraction
2) feature comparison / selection
3) model training

Each stage script remains runnable on its own:
- sl_dataframe_main.py
- sl_feature_comparison.py
- sl_training_pipeline.py

The feature-comparison stage is now split internally across:
- sl_feature_comparison.py
- sl_feature_comparison_tools.py
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable


@dataclass(frozen=True)
class Stage:
    # A small immutable description of one pipeline step.  Keeping the display
    # name together with its function makes it easy to run selected steps in
    # their intended order.
    name: str
    runner: Callable[[], None]


def _run_stage(stage_name: str, runner) -> None:
    # Wrap every stage with consistent logging and elapsed-time measurement.
    # perf_counter is used because it is intended for measuring durations.
    print("\n" + "=" * 80)
    print(f"[PIPELINE] Starting stage: {stage_name}")
    start = perf_counter()
    runner()  # Execute the main() function imported for this stage.
    elapsed = perf_counter() - start
    print(f"[PIPELINE] Finished stage: {stage_name} ({elapsed:.2f}s)")


def main(
    run_feature_extraction: bool = True,
    run_feature_analysis: bool = True,
    run_model_training: bool = True,
) -> None:
    # Build this list conditionally, so callers can run the complete pipeline
    # or only the stages they need (for example, retraining an existing setup).
    stages: list[Stage] = []

    if run_feature_extraction:
        # Import only when needed. This avoids loading a stage and its
        # dependencies when that stage was disabled by the caller.
        from sl_dataframe_main import main as dataframe_main

        stages.append(Stage("Feature Extraction", dataframe_main))

    if run_feature_analysis:
        # This stage compares/examines extracted features before training.
        from sl_feature_comparison import main as feature_comparison_main

        stages.append(Stage("Feature Comparison", feature_comparison_main))

    if run_model_training:
        # Training is last because it uses the outputs of preceding stages.
        from sl_training_pipeline import main as training_pipeline_main

        stages.append(Stage("Model Training", training_pipeline_main))

    if not stages:
        # Failing early gives a clear message instead of silently doing nothing.
        raise ValueError("No stages selected to run.")

    print("[PIPELINE] Shallow learning pipeline starting.")
    print(f"[PIPELINE] Selected stages: {[stage.name for stage in stages]}")

    # Measure the total time independently from the per-stage timings.
    pipeline_start = perf_counter()
    for stage in stages:
        _run_stage(stage.name, stage.runner)
    total_elapsed = perf_counter() - pipeline_start

    print("\n" + "=" * 80)
    print(f"[PIPELINE] All selected stages completed in {total_elapsed:.2f}s")


if __name__ == "__main__":
    # Run all stages when this file is executed directly. Imports of this module
    # can instead call main() with individual stages enabled or disabled.
    main()
