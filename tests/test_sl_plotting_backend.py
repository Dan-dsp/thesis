import os
from pathlib import Path
import subprocess
import sys
import unittest


class SLPlottingBackendTests(unittest.TestCase):
    def test_saved_plots_do_not_load_tk_before_threaded_training(self):
        for module in ("sl_training_pipeline", "sl_feature_comparison_tools"):
            with self.subTest(module=module):
                script = f'''
import gc
import importlib
from pathlib import Path
import sys
import tempfile
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

module = importlib.import_module({module!r})
plt = module._import_pyplot()
assert plt.get_backend().lower() == "agg", plt.get_backend()
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "plot.png"
    if {module!r} == "sl_training_pipeline":
        module.save_confusion_matrix(np.eye(2, dtype=int), np.array(["a", "b"]),
                                     path, "Test")
    else:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(path)
        plt.close(fig)
    assert path.read_bytes().startswith(b"\\x89PNG\\r\\n\\x1a\\n")
    assert not plt.get_fignums()
    X = np.arange(80).reshape(40, 2)
    y = np.array([0, 1] * 20)
    with joblib.parallel_backend("threading"):
        GridSearchCV(RandomForestClassifier(n_estimators=2, random_state=42),
                     {{"max_depth": [1, 2]}}, cv=2, n_jobs=2).fit(X, y)
        joblib.Parallel(n_jobs=2)(joblib.delayed(gc.collect)() for _ in range(4))
    assert "tkinter" not in sys.modules
'''
                env = dict(os.environ, MPLBACKEND="TkAgg")
                result = subprocess.run(
                    [sys.executable, "-c", script], env=env,
                    cwd=Path(__file__).resolve().parents[1], capture_output=True,
                    text=True, timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("Tcl_AsyncDelete", result.stderr)
                self.assertNotIn("main thread is not in main loop", result.stderr)


if __name__ == "__main__":
    unittest.main()
