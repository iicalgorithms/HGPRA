import os
import subprocess
import sys
import tempfile
import unittest


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from scripts.collect_results import collect_rows, latest_per_config, parse_log_file
from scripts.build_sensitivity_csv import build_rows as build_sensitivity_rows
from scripts.make_tables import render_latex_table
from scripts.merge_formal_ablation_results import build_rows as build_ablation_rows
from scripts.merge_formal_cross_arch_results import merge_rows as merge_cross_arch_rows
from scripts.merge_formal_main_results import merge_rows
from scripts.merge_formal_runtime_results import merge_rows as merge_runtime_rows
from scripts.run_experiment_grid import (
    build_experiments,
    build_commands,
    find_missing_modules,
    required_modules_for,
    write_shell_script,
)


class ExperimentScriptTests(unittest.TestCase):
    def test_ablation_preset_builds_expected_variants(self):
        experiments = build_experiments(
            preset="ablation",
            grid="cora:0.03",
            method="AllDeepSets",
            seed=15,
            log_root="logs_test",
            save_dir="save_test",
        )

        variants = {experiment.variant for experiment in experiments}
        self.assertEqual(variants, {"hgpra", "without_pas", "without_pra", "without_kged"})

        commands = build_commands(
            experiments=experiments,
            python_executable="/env/bin/python",
            stages=("buffer", "distill"),
            device="cuda:0",
            gpu_id=0,
            coreset_epochs=1,
            teacher_epochs=2,
            num_experts=1,
            traj_save_interval=1,
            param_save_interval=1,
            iterations=1,
            eval_interval=1,
            syn_steps=1,
            expert_epochs=1,
            min_start_epoch=1,
            max_start_epoch=10,
            max_start_epoch_s=2,
            nruns=1,
            test_model_iters=1,
            core_method="herding",
        )

        flat_commands = [" ".join(command) for command in commands]
        self.assertTrue(any("--difficulty_type random" in command for command in flat_commands))
        self.assertTrue(any("--expanding_window False" in command for command in flat_commands))
        self.assertTrue(any("--beta 0.0" in command for command in flat_commands))
        self.assertTrue(any("--max_start_epoch 10" in command for command in flat_commands))

    def test_buffer_command_is_deduplicated_across_ratios(self):
        experiments = build_experiments(
            preset="main",
            grid="cora:0.03,0.01",
            method="AllDeepSets",
            seed=15,
            log_root="logs_test",
            save_dir="save_test",
        )

        commands = build_commands(
            experiments=experiments,
            python_executable="/env/bin/python",
            stages=("buffer", "distill"),
            device="cuda:0",
            gpu_id=0,
            coreset_epochs=1,
            teacher_epochs=2,
            num_experts=1,
            traj_save_interval=1,
            param_save_interval=1,
            iterations=1,
            eval_interval=1,
            syn_steps=1,
            expert_epochs=1,
            min_start_epoch=1,
            max_start_epoch=10,
            max_start_epoch_s=2,
            nruns=1,
            test_model_iters=1,
            core_method="herding",
        )

        command_names = [command[1] for command in commands]
        self.assertEqual(command_names.count("buffer.py"), 1)
        self.assertEqual(command_names.count("distill.py"), 2)

    def test_sensitivity_can_reuse_artifact_commands_across_betas(self):
        experiments = build_experiments(
            preset="sensitivity",
            grid="cora:0.03",
            method="AllDeepSets",
            seed=15,
            log_root="logs_test",
            save_dir="save_test",
            sensitivity_values=(0.0, 0.1),
            reuse_artifacts_within_dataset=True,
        )

        commands = build_commands(
            experiments=experiments,
            python_executable="/env/bin/python",
            stages=("coreset", "buffer", "distill"),
            device="cuda:0",
            gpu_id=0,
            coreset_epochs=1,
            teacher_epochs=2,
            num_experts=1,
            traj_save_interval=1,
            param_save_interval=1,
            iterations=1,
            eval_interval=1,
            syn_steps=1,
            expert_epochs=1,
            min_start_epoch=1,
            max_start_epoch=10,
            max_start_epoch_s=2,
            nruns=1,
            test_model_iters=1,
            core_method="herding",
        )

        command_names = [command[1] for command in commands]
        flat_commands = [" ".join(command) for command in commands]
        self.assertEqual(command_names.count("utils/coreset.py"), 1)
        self.assertEqual(command_names.count("buffer.py"), 1)
        self.assertEqual(command_names.count("distill.py"), 2)
        self.assertTrue(all("--buffer_path logs_test/sensitivity/shared_artifacts/Buffer/cora-buffer" in command for command in flat_commands if "distill.py" in command))

    def test_write_shell_script_marks_file_executable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "run.sh")

            write_shell_script([["python", "buffer.py"]], script_path)

            self.assertTrue(os.access(script_path, os.X_OK))

    def test_parse_log_file_extracts_best_accuracy_and_elapsed_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "ablation", "hgpra", "Distill", "cora-reduce_0.03-20260525")
            os.makedirs(log_dir)
            log_path = os.path.join(log_dir, "train.log")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("05/25 01:00:00 PM args = Namespace(dname='cora', method='AllDeepSets', beta=0.1, seed=15)\n")
                handle.write("05/25 01:00:05 PM new best test_acc occurs: eval_acc = 65.0000, test_acc = 66.5000, iteration = 1\n")
                handle.write("05/25 01:00:10 PM Evaluation ACC: AllDeepSets best test_acc = 0.66500, best_iter = 1\n")

            row = parse_log_file(log_path)

        self.assertEqual(row["dataset"], "cora")
        self.assertEqual(row["ratio"], "0.03")
        self.assertEqual(row["preset"], "ablation")
        self.assertEqual(row["variant"], "hgpra")
        self.assertAlmostEqual(float(row["test_acc_percent"]), 66.5)
        self.assertEqual(row["best_iter"], "1")
        self.assertEqual(row["elapsed_seconds"], "10")

    def test_parse_log_file_accepts_iso_timestamps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "ablation", "hgpra", "Distill", "cora-reduce_0.03-20260525")
            os.makedirs(log_dir)
            log_path = os.path.join(log_dir, "train.log")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("2026-05-25 13:00:00,001 args = Namespace(dname='cora', method='AllDeepSets')\n")
                handle.write("2026-05-25 13:00:02,999 Evaluation ACC: AllDeepSets best test_acc = 0.66500, best_iter = 1\n")

            row = parse_log_file(log_path)

        self.assertEqual(row["elapsed_seconds"], "2")

    def test_parse_log_file_keeps_std_from_best_mean(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "main", "hgpra", "Distill", "cora-reduce_0.03-20260525")
            os.makedirs(log_dir)
            log_path = os.path.join(log_dir, "train.log")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("2026-05-25 13:00:00,001 args = Namespace(dname='cora', method='AllDeepSets')\n")
                handle.write("2026-05-25 13:00:01,001 TEST: Full Graph Mean Accuracy: 0.650000, STD: 0.010000\n")
                handle.write("2026-05-25 13:00:01,002 new best test_acc occurs: eval_acc = 66.0000, test_acc = 65.0000, iteration = 4\n")
                handle.write("2026-05-25 13:00:02,001 TEST: Full Graph Mean Accuracy: 0.600000, STD: 0.090000\n")

            row = parse_log_file(log_path)

        self.assertEqual(row["test_acc_percent"], "65.000000")
        self.assertEqual(row["test_std_percent"], "1.000000")
        self.assertEqual(row["best_iter"], "4")

    def test_collect_rows_can_ignore_incomplete_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            complete_dir = os.path.join(tmpdir, "main", "hgpra", "Distill", "cora-reduce_0.03-complete")
            incomplete_dir = os.path.join(tmpdir, "main", "hgpra", "Distill", "cora-reduce_0.01-incomplete")
            os.makedirs(complete_dir)
            os.makedirs(incomplete_dir)
            with open(os.path.join(complete_dir, "train.log"), "w", encoding="utf-8") as handle:
                handle.write("2026-05-25 13:00:00,001 args = Namespace(dname='cora', method='AllDeepSets', ITER=100)\n")
                handle.write("2026-05-25 13:00:01,001 Iteration 100: Total_Loss = 0.99\n")
            with open(os.path.join(incomplete_dir, "train.log"), "w", encoding="utf-8") as handle:
                handle.write("2026-05-25 13:00:00,001 args = Namespace(dname='cora', method='AllDeepSets', ITER=100)\n")
                handle.write("2026-05-25 13:00:01,001 Iteration 14: Total_Loss = 0.99\n")

            rows = collect_rows(tmpdir, complete_only=True)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ratio"], "0.03")

    def test_latest_per_config_keeps_newest_log_path(self):
        rows = [
            {
                "dataset": "cora",
                "ratio": "0.03",
                "preset": "ablation",
                "variant": "hgpra",
                "method": "AllDeepSets",
                "seed": "15",
                "beta": "0.1",
                "expanding_window": "True",
                "log_path": "run-20260528-000000/train.log",
            },
            {
                "dataset": "cora",
                "ratio": "0.03",
                "preset": "ablation",
                "variant": "hgpra",
                "method": "AllDeepSets",
                "seed": "15",
                "beta": "0.1",
                "expanding_window": "True",
                "log_path": "run-20260528-010000/train.log",
            },
        ]

        latest = latest_per_config(rows)

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["log_path"], "run-20260528-010000/train.log")

    def test_merge_formal_main_results_updates_paper_hgpra_column(self):
        paper_rows = [
            {
                "dataset": "Cora-CA",
                "ratio": "3.00%",
                "Random": "1.00+-0.00",
                "Herding": "1.00+-0.00",
                "K-Center": "1.00+-0.00",
                "HyperSF": "1.00+-0.00",
                "HyperEF": "1.00+-0.00",
                "HG-Cond": "1.00+-0.00",
                "HGPRA": "old",
                "Full Training Set": "2.00+-0.00",
            }
        ]
        formal_rows = [
            {
                "dataset": "coauthor_cora",
                "ratio": "0.03",
                "preset": "main",
                "variant": "hgpra",
                "method": "AllDeepSets",
                "test_acc_percent": "62.275000",
                "test_std_percent": "3.047400",
            }
        ]

        updated = merge_rows(paper_rows, formal_rows, require_all=True)

        self.assertEqual(updated, 1)
        self.assertEqual(paper_rows[0]["HGPRA"], "62.28+-3.05")

    def test_merge_formal_ablation_results_builds_component_table(self):
        formal_rows = [
            {
                "dataset": "coauthor_cora",
                "ratio": "0.03",
                "preset": "ablation",
                "variant": "hgpra",
                "method": "AllDeepSets",
                "test_acc_percent": "62.275000",
                "test_std_percent": "3.047400",
            },
            {
                "dataset": "coauthor_cora",
                "ratio": "0.03",
                "preset": "ablation",
                "variant": "without_pas",
                "method": "AllDeepSets",
                "test_acc_percent": "60.000000",
                "test_std_percent": "1.000000",
            },
            {
                "dataset": "coauthor_cora",
                "ratio": "0.03",
                "preset": "ablation",
                "variant": "without_pra",
                "method": "AllDeepSets",
                "test_acc_percent": "59.000000",
                "test_std_percent": "1.000000",
            },
            {
                "dataset": "coauthor_cora",
                "ratio": "0.03",
                "preset": "ablation",
                "variant": "without_kged",
                "method": "AllDeepSets",
                "test_acc_percent": "58.000000",
                "test_std_percent": "1.000000",
            },
        ]

        rows = build_ablation_rows(formal_rows)

        self.assertEqual(rows[0]["setting"], "Full HGPRA")
        self.assertEqual(rows[0]["Cora-CA"], "62.28+-3.05")
        self.assertEqual(rows[0]["Citeseer-CC"], "")

    def test_merge_formal_runtime_results_updates_hgpra_column(self):
        paper_rows = [{"dataset": "Cora-CA", "ratio": "3%", "HG-Cond": "23s", "HGPRA": "old"}]
        formal_rows = [
            {
                "dataset": "coauthor_cora",
                "ratio": "0.03",
                "variant": "hgpra",
                "method": "AllDeepSets",
                "elapsed_seconds": "125",
            }
        ]

        updated = merge_runtime_rows(paper_rows, formal_rows, require_all=True)

        self.assertEqual(updated, 1)
        self.assertEqual(paper_rows[0]["HGPRA"], "2m 5s")

    def test_merge_formal_cross_arch_results_updates_hgpra_row(self):
        paper_rows = [
            {
                "dataset": "Cora-CA",
                "ratio": "3%",
                "method": "HGPRA",
                "HyperGCN": "old",
                "HGNN": "old",
                "AllDeepSets": "old",
                "AllSetTransformer": "old",
            }
        ]
        formal_rows = [
            {
                "dataset": "coauthor_cora",
                "ratio": "0.03",
                "preset": "main",
                "variant": "hgpra",
                "method": "HGNN",
                "test_acc_percent": "58.114000",
            }
        ]

        updated = merge_cross_arch_rows(paper_rows, formal_rows, require_all=False)

        self.assertEqual(updated, 1)
        self.assertEqual(paper_rows[0]["HGNN"], "58.11")

    def test_build_sensitivity_csv_keeps_beta_curve_rows(self):
        formal_rows = [
            {
                "dataset": "20newsW100",
                "ratio": "0.01",
                "method": "AllDeepSets",
                "beta": "0.1",
                "test_acc_percent": "74.292000",
                "test_std_percent": "0.791700",
                "log_path": "main/train.log",
            },
            {
                "dataset": "ModelNet40",
                "ratio": "0.01",
                "method": "AllDeepSets",
                "beta": "0.01",
                "test_acc_percent": "92.000000",
                "test_std_percent": "0.500000",
                "log_path": "sensitivity/train.log",
            },
            {
                "dataset": "coauthor_cora",
                "ratio": "0.03",
                "method": "AllDeepSets",
                "beta": "0.1",
                "test_acc_percent": "62.000000",
            },
        ]

        rows = build_sensitivity_rows(formal_rows)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["dataset"], "20News")
        self.assertEqual(rows[0]["beta"], "0.1")
        self.assertEqual(rows[1]["dataset"], "ModelNet40")

    def test_render_latex_table_aggregates_repeated_rows(self):
        rows = [
            {"dataset": "cora", "ratio": "0.03", "method": "AllDeepSets", "variant": "hgpra", "test_acc_percent": "66.0"},
            {"dataset": "cora", "ratio": "0.03", "method": "AllDeepSets", "variant": "hgpra", "test_acc_percent": "68.0"},
        ]

        rendered = render_latex_table(rows, caption="Ablation summary.", label="tab:ablation_generated")

        self.assertIn("Ablation summary.", rendered)
        self.assertIn("cora", rendered)
        self.assertIn("67.00 $\\pm$ 1.00", rendered)

    def test_dependency_preflight_reports_missing_modules(self):
        self.assertEqual(find_missing_modules(sys.executable, ["json"]), [])
        missing = find_missing_modules(sys.executable, ["hgpra_missing_module_for_test"])
        self.assertEqual(len(missing), 1)
        self.assertTrue(missing[0].startswith("hgpra_missing_module_for_test: ModuleNotFoundError:"))

    def test_alldeepsets_preflight_does_not_require_optional_backends(self):
        modules = required_modules_for(methods=["AllDeepSets"], stages=("coreset", "buffer", "distill"))
        self.assertIn("torch", modules)
        self.assertIn("torch_geometric", modules)
        self.assertNotIn("dhg", modules)
        self.assertNotIn("torch_sparse", modules)

    def test_default_imports_do_not_require_optional_backends(self):
        probe = (
            "import sys; "
            f"sys.path.insert(0, {CODE_DIR!r}); "
            "import utils.dataloader; "
            "import model.model_loader; "
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
