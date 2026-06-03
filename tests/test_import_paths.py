import os
import subprocess
import sys
import unittest


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class ImportPathTests(unittest.TestCase):
    def test_coreset_direct_script_can_find_preprocessing_module(self):
        result = subprocess.run(
            [sys.executable, "utils/coreset.py", "--help"],
            cwd=CODE_DIR,
            env={**os.environ, "OMP_NUM_THREADS": "1", "KMP_INIT_AT_FORK": "FALSE"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertNotIn("No module named 'preprocessing'", result.stderr)
        self.assertNotIn("'utils' is not a package", result.stderr)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_setgnn_wrapper_imports_as_package_module(self):
        probe = (
            "import sys; "
            f"sys.path.insert(0, {CODE_DIR!r}); "
            "import model.setgnn_wrapper; "
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            env={**os.environ, "OMP_NUM_THREADS": "1", "KMP_INIT_AT_FORK": "FALSE"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
