"""Run the existing thesis scripts sequentially; default: refresh SL and reuse DL."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
CHECKPOINT_NAME = "bird_species_resnet18_FINAL_best.pth"


@dataclass(frozen=True)
class Stage:
    name: str
    arguments: tuple[str, ...]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--full", action="store_true",
                       help="Also extract SL features and retrain both DL configurations.")
    modes.add_argument("--reports-only", action="store_true",
                       help="Reuse existing models; run comparison and image-error reporting only.")
    parser.add_argument("--with-statistics", action="store_true",
                        help="Also run nested SL CV and within-family SL/DL statistical analysis.")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without running them.")
    parser.add_argument("--dl-full-model", type=Path,
                        help="Existing full-DL checkpoint for default/report-only mode.")
    parser.add_argument("--dl-partial-model", type=Path,
                        help="Existing partial-DL checkpoint for default/report-only mode.")
    args = parser.parse_args(argv)
    if args.full and (args.dl_full_model or args.dl_partial_model):
        parser.error("--full uses newly trained checkpoints; do not override them with old models.")
    if args.reports_only and args.with_statistics:
        parser.error("Nested CV trains models; omit --reports-only to use --with-statistics.")
    return args


def checkpoint_paths(args, root=ROOT):
    if args.full:
        full = root / "outputs_resnet18_thesis" / "final_train" / CHECKPOINT_NAME
    else:
        # Explicitly reuse the historical bbox checkpoint, never the no-bbox experiment.
        full = args.dl_full_model or root / "outputs_resnet18_thesis_dl_bbox" / "final_train" / CHECKPOINT_NAME
    partial = (args.dl_partial_model or
               root / "outputs_resnet18_thesis_partial_finetune" / "final_train" / CHECKPOINT_NAME)
    return tuple((root / path).resolve() if not path.is_absolute() else path.resolve()
                 for path in (full, partial))


def build_stages(args, root=ROOT):
    full, partial = checkpoint_paths(args, root)
    stages = []
    if args.full:
        stages.extend([
            Stage("SL feature extraction", ("sl_dataframe_main.py",)),
            Stage("DL full fine-tuning and CV", ("dl_transfer_learning_bird_classifier.py",)),
            Stage("DL partial fine-tuning and CV", ("dl_transfer_learning_bird_classifier_partial_finetune.py",)),
        ])
    if not args.reports_only:
        stages.extend([
            Stage("SL train-only feature selection", ("sl_feature_comparison.py",)),
            Stage("SL selected-feature batch training", ("sl_training_pipeline.py", "--run-exported-batch")),
        ])
    if args.with_statistics:
        sl_cv = root.parent / "sl_nested_cv"
        stages.extend([
            Stage("SL independent nested CV (MI top-50)", (
                "thesis_validation.py", "sl-cv", "--method", "mi", "--features", "50",
                "--out-dir", str(sl_cv))),
            Stage("SL within-family statistics", (
                "thesis_validation.py", "compare", str(sl_cv / "outer_cv_scores.csv"),
                "--out-dir", str(root.parent / "sl_statistics"))),
            Stage("DL within-family statistics", (
                "thesis_validation.py", "compare",
                str(full.parent.parent / "kfold_train_only" / "outer_cv_scores.csv"),
                str(partial.parent.parent / "kfold_train_only" / "outer_cv_scores.csv"),
                "--out-dir", str(root.parent / "dl_statistics"))),
        ])
    stages.append(Stage("Family reports and shared-image error gallery", (
        "compare_dl_sl_models.py", "--dl-full-model", str(full), "--dl-partial-model", str(partial))))
    return stages


def preflight(args, stages, root=ROOT):
    for stage in stages:
        if not (root / stage.arguments[0]).is_file():
            raise FileNotFoundError(f"Missing pipeline script: {stage.arguments[0]}")
    full, partial = checkpoint_paths(args, root)
    if not args.full:
        for path in (full, partial):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing saved DL checkpoint: {path}. Use --full to retrain, "
                    "or specify --dl-full-model / --dl-partial-model.")
    if not args.reports_only:
        if not args.full:
            features = root.parent / "dataset_features" / "shallow_learning_birds.csv"
            with features.open(newline="", encoding="utf-8-sig") as handle:
                columns = next(csv.reader(handle))
            if not {f"fourier_{i}" for i in range(1, 21)}.issubset(columns):
                raise ValueError("Saved feature CSV lacks Fourier columns; use --full or rerun extraction.")
        else:
            for split in ("train", "test"):
                path = root.parent / "dataset_bbox" / split
                if not path.is_dir():
                    raise FileNotFoundError(f"Prepare cropped images first: {path}")
    if args.with_statistics and not args.full:
        for checkpoint in (full, partial):
            scores = checkpoint.parent.parent / "kfold_train_only" / "outer_cv_scores.csv"
            with scores.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            if not rows or any(row.get("evaluation_role") != "independent_outer" for row in rows):
                raise ValueError(f"Independent DL CV scores required: {scores}. Use --full --with-statistics.")
    if args.reports_only and not (root.parent / "sl_outputs_batch" / "latest_batch_run.txt").is_file():
        raise FileNotFoundError("No saved SL batch pointer. Run the default pipeline first.")


def _run_stage(command, root, log_path):
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    encoding = sys.stdout.encoding or "utf-8"
                    sys.stdout.write(line.encode(encoding, errors="replace").decode(encoding))
                    sys.stdout.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def execute(stages, root=ROOT):
    logs = root / "pipeline_runs" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    logs.mkdir(parents=True, exist_ok=False)
    status = {"python": sys.executable, "working_directory": str(root), "stages": []}
    status_path = logs / "status.json"
    for i, stage in enumerate(stages, 1):
        command = [sys.executable, "-u", *stage.arguments]
        log = logs / f"{i:02d}_{Path(stage.arguments[0]).stem}.log"
        print(f"\n[PIPELINE {i}/{len(stages)}] {stage.name}", flush=True)
        print(f"[LOG] {log}", flush=True)
        record = {"name": stage.name, "command": command, "log": str(log), "status": "running"}
        status["stages"].append(record)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        start = time.perf_counter()
        try:
            _run_stage(command, root, log)
        except (subprocess.CalledProcessError, KeyboardInterrupt, OSError) as exc:
            record.update(status="failed", error=str(exc))
            raise
        else:
            record["status"] = "completed"
        finally:
            record["seconds"] = time.perf_counter() - start
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return logs


def main(argv=None):
    args = parse_args(argv)
    stages = build_stages(args)
    print(f"Python: {sys.executable}\nWorking directory: {ROOT}")
    for i, stage in enumerate(stages, 1):
        print(f"{i}. {stage.name}\n   {subprocess.list2cmdline([sys.executable, '-u', *stage.arguments])}")
    if args.dry_run:
        print("Preview only: no scripts executed or files modified.")
        return 0
    try:
        preflight(args, stages)
        logs = execute(stages)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"\n[STOPPED] {exc}\nNo later stages were started.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[STOPPED] Interrupted; no later stages were started.", file=sys.stderr)
        return 130
    gallery = ROOT / "family_evaluation_results" / "image_error_analysis" / "error_gallery.html"
    print(f"\n[COMPLETE] Logs: {logs}\nOpen the local HTML report: {gallery}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
