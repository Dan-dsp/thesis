from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_pipeline as runner


class PipelineRunnerTests(unittest.TestCase):
    def test_default_reuses_dl_and_refreshes_selected_sl_features(self):
        args = runner.parse_args([])
        stages = runner.build_stages(args)
        self.assertEqual([s.arguments[0] for s in stages], ["sl_feature_comparison.py",
                         "sl_training_pipeline.py", "compare_dl_sl_models.py"])
        self.assertIn("--run-exported-batch", stages[1].arguments)
        self.assertIn("outputs_resnet18_thesis_dl_bbox", stages[-1].arguments[2])

    def test_full_mode_uses_new_dl_outputs(self):
        stages = runner.build_stages(runner.parse_args(["--full"]))
        self.assertEqual(len(stages), 6)
        self.assertEqual(stages[0].arguments, ("sl_dataframe_main.py",))
        self.assertEqual(stages[1].arguments, ("dl_transfer_learning_bird_classifier.py",))
        self.assertIn("outputs_resnet18_thesis", stages[-1].arguments[2])
        self.assertNotIn("outputs_resnet18_thesis_dl_bbox", stages[-1].arguments[2])

    def test_reports_only_and_explicit_checkpoint(self):
        stages = runner.build_stages(runner.parse_args([
            "--reports-only", "--dl-full-model", "custom/full.pth"]))
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0].arguments[0], "compare_dl_sl_models.py")
        self.assertEqual(Path(stages[0].arguments[2]), (runner.ROOT / "custom/full.pth").resolve())

    def test_statistics_remain_separate_within_families(self):
        stages = runner.build_stages(runner.parse_args(["--full", "--with-statistics"]))
        self.assertEqual(len(stages), 9)
        statistical = [s for s in stages if s.arguments[0] == "thesis_validation.py"]
        self.assertEqual(len(statistical), 3)
        self.assertEqual(statistical[0].arguments[1], "sl-cv")
        self.assertIn("sl_nested_cv", statistical[1].arguments[2])
        self.assertTrue(all("resnet18" in arg for arg in statistical[2].arguments[2:4]))

    def test_incompatible_modes_fail(self):
        for arguments in [["--full", "--reports-only"],
                          ["--reports-only", "--with-statistics"],
                          ["--full", "--dl-full-model", "old.pth"]]:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.parse_args(arguments)

    def test_dry_run_never_checks_inputs_or_executes(self):
        with patch.object(runner, "preflight") as check, patch.object(runner, "execute") as execute, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(["--dry-run"]), 0)
            check.assert_not_called()
            execute.assert_not_called()

    def test_missing_checkpoint_is_detected_before_any_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parse_args(["--dl-full-model", str(Path(tmp) / "missing.pth")])
            with self.assertRaisesRegex(FileNotFoundError, "Missing saved DL checkpoint"):
                runner.preflight(args, runner.build_stages(args))

    def test_execution_stops_and_saves_failure_status(self):
        calls = []

        def child(command, root, log):
            calls.append(command)
            log.write_text("captured output", encoding="utf-8")
            if len(calls) == 2:
                raise subprocess.CalledProcessError(2, command)

        stages = [runner.Stage(name, (name + ".py",)) for name in ["one", "two", "three"]]
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "_run_stage", side_effect=child), \
                redirect_stdout(io.StringIO()):
            root = Path(tmp)
            with self.assertRaises(subprocess.CalledProcessError):
                runner.execute(stages, root)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0], sys.executable)
            status_file = next(root.glob("pipeline_runs/*/status.json"))
            status = json.loads(status_file.read_text())
            self.assertEqual([s["status"] for s in status["stages"]], ["completed", "failed"])

    def test_child_output_is_logged_and_nonzero_exit_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            log = root / "child.log"
            runner._run_stage([sys.executable, "-u", "-c", "import os; print(os.getcwd())"], root, log)
            self.assertEqual(Path(log.read_text().strip()), root)
            with self.assertRaises(subprocess.CalledProcessError):
                runner._run_stage([sys.executable, "-u", "-c", "print('failure detail'); raise SystemExit(3)"], root, log)
            self.assertIn("failure detail", log.read_text())


if __name__ == "__main__":
    unittest.main()
