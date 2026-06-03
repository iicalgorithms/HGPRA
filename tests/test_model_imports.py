import os
import sys
import tempfile
import unittest


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


class ModelImportTests(unittest.TestCase):
    def test_hypergcn_wrapper_uses_package_relative_preprocessing_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                from model.hypergcn_wrapper import HyperGCNWrapper
            finally:
                if old_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = old_home

        self.assertEqual(HyperGCNWrapper.__name__, "HyperGCNWrapper")


if __name__ == "__main__":
    unittest.main()
