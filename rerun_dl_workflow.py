"""Retrain both DL variants and refresh selection summaries and SL-versus-DL reports.

Run: python rerun_dl_workflow.py
Preview: python rerun_dl_workflow.py --dry-run
Existing SL models/features and the thesis source are not retrained/edited.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from run_pipeline import Stage, execute


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
CHECKPOINT = "bird_species_resnet18_FINAL_best.pth"
MODELS = ("knn", "random_forest", "svm_rbf", "xgboost")


def build_stages(run_dir, sl_run_dir, reports_only=False):
    run_dir = Path(run_dir).resolve()
    full = run_dir / "full"
    partial = run_dir / "partial"
    stages = []
    if not reports_only:
        stages.extend([
            Stage("Full DL: five 3+5 fold runs and mean-epoch refit", (
                "dl_transfer_learning_bird_classifier.py", "--output-dir", str(full))),
            Stage("Partial DL: five 3+5 fold runs and mean-epoch refit", (
                "dl_transfer_learning_bird_classifier_partial_finetune.py", "--output-dir", str(partial))),
        ])
    stages.append(Stage("Descriptive DL selection comparison (not outer-CV inference)", (
        "dl_cv_workflow.py", str(full / "cv_selection" / "fold_scores.csv"),
        str(partial / "cv_selection" / "fold_scores.csv"),
        "--out-dir", str(run_dir / "dl_selection_comparison"))))
    for balanced, folder in ((True, "comparison_balanced"), (False, "comparison_full_test")):
        stages.append(Stage(f"SL vs DL: {'balanced' if balanced else 'full'} test and error gallery", (
            "compare_dl_sl_models.py", "--dl-full-model", str(full / "final_train" / CHECKPOINT),
            "--dl-partial-model", str(partial / "final_train" / CHECKPOINT),
            "--sl-run-dir", str(Path(sl_run_dir).resolve()),
            "--balance-test" if balanced else "--no-balance-test",
            "--output-dir", str(run_dir / folder))))
    return stages


def resolve_sl_run(explicit=None, project=PROJECT):
    if explicit:
        return Path(explicit).resolve()
    pointer = project / "sl_outputs_batch" / "latest_batch_run.txt"
    batch = Path(pointer.read_text(encoding="utf-8-sig").strip())
    if not batch.is_absolute():
        batch = project / batch
    selected = json.loads((batch / "selected_model.json").read_text(encoding="utf-8-sig"))
    model = Path(selected["model_path"])
    if not model.is_absolute():
        model = batch / model
    return model.parent.parent.resolve()


def preflight(run_dir, sl_run, stages, reports_only, root=ROOT):
    for stage in stages:
        if not (root / stage.arguments[0]).is_file():
            raise FileNotFoundError(f"Missing script: {stage.arguments[0]}")
    for name in MODELS:
        path = sl_run / "models" / f"{name}_final.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Saved SL model is missing: {path}")
    for filename in ("feature_columns.json", "class_names.json", "label_encoder.pkl"):
        if not (sl_run / "metadata" / filename).is_file():
            raise FileNotFoundError(f"Missing saved SL metadata: {filename}")
    if not (root.parent / "dataset_features" / "test").is_dir():
        raise FileNotFoundError("Missing saved SL test descriptors")
    # Reuse, do not regenerate pixels underneath already-trained SL models.
    from shared_augmentation import load_augmentation
    _, augmentation = load_augmentation()
    split_info = json.loads((sl_run / "metadata" / "split_summary.json").read_text())
    for identity in ("training_pool_id", "augmentation_id"):
        if split_info.get(identity) != augmentation.get(identity):
            raise ValueError(f"Saved SL and current DL data disagree on {identity}. "
                             "Do not compare stale SL models; rebuild SL for the current data first.")
    for split in ("train", "test"):
        if not (root.parent / "dataset_bbox" / split).is_dir():
            raise FileNotFoundError(f"Missing cropped-image dataset: {split}")
    if reports_only:
        for variant in ("full", "partial"):
            base = run_dir / variant
            for path in (base / "final_train" / CHECKPOINT, base / "cv_selection" / "fold_scores.csv"):
                if not path.is_file():
                    raise FileNotFoundError(f"Run training first; missing {path}")
            protocol = json.loads((base / "final_train" / "refit_protocol.json").read_text())
            if protocol.get("validation_protocol") != "pooled_rows_five_fold_mean_epochs_v1":
                raise ValueError(f"Not a mean-epoch CV refit: {base}")
    elif run_dir.exists():
        raise ValueError("Choose a NEW --run-dir for training; existing results will not be overwritten. "
                         "Use --reports-only --run-dir PATH to rebuild reports from a completed run.")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-dir", type=Path, help="New run directory, or completed run with --reports-only")
    parser.add_argument("--reports-only", action="store_true", help="Reuse models in the specified completed run")
    parser.add_argument("--sl-run-dir", type=Path, help="Override the saved SL batch selection")
    args = parser.parse_args(argv)
    if args.reports_only and not args.run_dir:
        parser.error("--reports-only requires --run-dir from the completed training run")
    run_dir = (args.run_dir or PROJECT / "dl_workflow_runs" /
               datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")).resolve()
    try:
        # Reuse the same SL run when repairing reports, even if the batch pointer changed.
        manifest_path = run_dir / "workflow_manifest.json"
        if args.reports_only and not args.sl_run_dir and manifest_path.is_file():
            sl_run = Path(json.loads(manifest_path.read_text())["sl_run_dir"])
        else:
            sl_run = resolve_sl_run(args.sl_run_dir)
        stages = build_stages(run_dir, sl_run, args.reports_only)
        print(f"Results: {run_dir}\nReusing SL models: {sl_run}")
        for i, stage in enumerate(stages, 1):
            print(f"{i}. {stage.name}\n   {subprocess.list2cmdline([sys.executable, '-u', *stage.arguments])}")
        if args.dry_run:
            print("Preview only. No training, reports, or files changed.")
            return 0
        preflight(run_dir, sl_run, stages, args.reports_only)
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"status": "running", "run_dir": str(run_dir), "sl_run_dir": str(sl_run),
                    "protocol": "pooled_rows_five_fold_mean_epochs_v1", "python": sys.executable,
                    "reports_only": args.reports_only,
                    "stages": [{"name": stage.name, "arguments": stage.arguments} for stage in stages]}
        manifest["source_sha256"] = {name: file_hash(ROOT / name) for name in (
            "dl_cv_workflow.py", "dl_training_protocol.py", "rerun_dl_workflow.py",
            "dl_transfer_learning_bird_classifier.py",
            "dl_transfer_learning_bird_classifier_partial_finetune.py", "compare_dl_sl_models.py")}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        try:
            logs = execute(stages, root=ROOT)
        except BaseException:
            manifest["status"] = "failed_or_interrupted"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            raise
        checkpoints = [run_dir / variant / "final_train" / CHECKPOINT for variant in ("full", "partial")]
        manifest.update(status="complete", logs=str(logs),
                        checkpoint_sha256={str(path): file_hash(path) for path in checkpoints})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        pointer = PROJECT / "dl_workflow_runs" / "latest_run.txt"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(str(run_dir), encoding="utf-8")
        print(f"\n[COMPLETE] Results: {run_dir}\nLogs: {logs}")
        print(f"Balanced gallery: {run_dir / 'comparison_balanced' / 'image_error_analysis' / 'error_gallery.html'}")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"[STOPPED] {exc}\nNo later stages were started.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[STOPPED] Interrupted; no later stages were started.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
