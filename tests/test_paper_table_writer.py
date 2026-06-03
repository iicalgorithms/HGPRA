import csv
import os
import sys
import tempfile
import unittest


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from scripts.write_paper_tables import write_main_performance_table, write_runtime_table


class PaperTableWriterTests(unittest.TestCase):
    def test_write_main_performance_table_marks_best_condensed_result(self):
        rows = [
            {
                "dataset": "Cora-CA",
                "ratio": "3.00%",
                "Random": "42.75±0.30",
                "Herding": "42.51±2.92",
                "K-Center": "45.54±4.00",
                "HyperSF": "41.61±5.00",
                "HyperEF": "42.79±4.59",
                "HG-Cond": "56.28±0.52",
                "HGPRA": "66.59±1.67",
                "Full Training Set": "77.18±1.98",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "main.csv")
            out_path = os.path.join(tmpdir, "table.tex")
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            write_main_performance_table(csv_path, out_path)

            with open(out_path, "r", encoding="utf-8") as handle:
                rendered = handle.read()

        self.assertIn("\\label{tab:performance_comparison}", rendered)
        self.assertIn("\\textbf{\\boldmath $66.59\\pm1.67$}", rendered)
        self.assertIn("Full Training Set", rendered)
        self.assertIn("\\multirow{1}{*}{Cora-CA} & 3.00\\%", rendered)

    def test_write_runtime_table_keeps_multirow_on_same_latex_row(self):
        rows = [
            {"dataset": "Cora-CA", "ratio": "3%", "HG-Cond": "23s", "HGPRA": "14s"},
            {"dataset": "Cora-CA", "ratio": "1%", "HG-Cond": "23s", "HGPRA": "13s"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "runtime.csv")
            out_path = os.path.join(tmpdir, "time.tex")
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            write_runtime_table(csv_path, out_path)

            with open(out_path, "r", encoding="utf-8") as handle:
                rendered = handle.read()

        self.assertIn("\\multirow{2}{*}{Cora-CA} & 3\\% & 23s & 14s", rendered)


if __name__ == "__main__":
    unittest.main()
