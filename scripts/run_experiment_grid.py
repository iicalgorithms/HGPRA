#!/usr/bin/env python3
"""Generate and optionally run HGPRA experiment grids."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


BASE_CHECK_MODULES = (
    "numpy",
    "scipy",
    "sklearn",
    "torch",
    "torch_geometric",
    "torch_scatter",
    "tqdm",
)

METHOD_EXTRA_CHECK_MODULES = {
    "HyperGCN": ("dhg",),
    "UniGCNII": ("torch_sparse", "torch_scatter"),
}

DEFAULT_CHECK_MODULES = BASE_CHECK_MODULES


@dataclass(frozen=True)
class Experiment:
    preset: str
    variant: str
    dataset: str
    ratio: str
    method: str
    seed: int
    log_root: str
    save_dir: str
    difficulty_type: str = "node"
    expanding_window: bool = True
    beta: float = 0.1
    soft_label: bool = False
    artifact_save_log: str = ""

    @property
    def save_log(self) -> str:
        return os.path.join(self.log_root, self.preset, self.variant)

    @property
    def artifact_log(self) -> str:
        return self.artifact_save_log or self.save_log


def parse_grid(grid: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for dataset_spec in grid.split(";"):
        dataset_spec = dataset_spec.strip()
        if not dataset_spec:
            continue
        if ":" not in dataset_spec:
            raise ValueError(f"Grid entry must be DATASET:RATIO[,RATIO], got {dataset_spec!r}")
        dataset, ratios = dataset_spec.split(":", 1)
        dataset = dataset.strip()
        for ratio in ratios.split(","):
            ratio = ratio.strip()
            if ratio:
                pairs.append((dataset, ratio))
    if not pairs:
        raise ValueError("No dataset-ratio pairs found in grid")
    return pairs


def _variant_configs(preset: str, sensitivity_values: Sequence[float]) -> List[dict]:
    if preset in {"main", "efficiency"}:
        return [{"variant": "hgpra"}]
    if preset == "ablation":
        return [
            {"variant": "hgpra"},
            {"variant": "without_pas", "difficulty_type": "random"},
            {"variant": "without_pra", "expanding_window": False},
            {"variant": "without_kged", "beta": 0.0},
        ]
    if preset == "sensitivity":
        return [{"variant": f"beta_{value:g}", "beta": float(value)} for value in sensitivity_values]
    raise ValueError(f"Unsupported preset {preset!r}")


def build_experiments(
    preset: str,
    grid: str,
    method: str,
    seed: int,
    log_root: str,
    save_dir: str,
    sensitivity_values: Sequence[float] = (0.0, 0.01, 0.05, 0.1, 0.2),
    reuse_artifacts_within_dataset: bool = False,
) -> List[Experiment]:
    experiments: List[Experiment] = []
    for dataset, ratio in parse_grid(grid):
        for config in _variant_configs(preset, sensitivity_values):
            variant = config.get("variant", "hgpra")
            artifact_save_log = ""
            if reuse_artifacts_within_dataset:
                if preset == "sensitivity":
                    artifact_save_log = os.path.join(log_root, preset, "shared_artifacts")
                elif preset == "ablation":
                    group = "random_artifacts" if config.get("difficulty_type") == "random" else "node_artifacts"
                    artifact_save_log = os.path.join(log_root, preset, group)
            experiments.append(
                Experiment(
                    preset=preset,
                    variant=variant,
                    dataset=dataset,
                    ratio=ratio,
                    method=method,
                    seed=seed,
                    log_root=log_root,
                    save_dir=save_dir,
                    difficulty_type=config.get("difficulty_type", "node"),
                    expanding_window=config.get("expanding_window", True),
                    beta=float(config.get("beta", 0.1)),
                    soft_label=bool(config.get("soft_label", False)),
                    artifact_save_log=artifact_save_log,
                )
            )
    return experiments


def required_modules_for(methods: Sequence[str], stages: Sequence[str]) -> Tuple[str, ...]:
    modules = list(BASE_CHECK_MODULES)
    for method in methods:
        for module in METHOD_EXTRA_CHECK_MODULES.get(method, ()):
            if module not in modules:
                modules.append(module)
    return tuple(modules)


def _bool_text(value: bool) -> str:
    return "True" if value else "False"


def build_commands(
    experiments: Sequence[Experiment],
    python_executable: str,
    stages: Sequence[str],
    device: str,
    gpu_id: int,
    coreset_epochs: int,
    teacher_epochs: int,
    num_experts: int,
    traj_save_interval: int,
    param_save_interval: int,
    iterations: int,
    eval_interval: int,
    syn_steps: int,
    expert_epochs: int,
    min_start_epoch: int,
    max_start_epoch: int,
    max_start_epoch_s: int,
    nruns: int,
    test_model_iters: int,
    core_method: str,
) -> List[List[str]]:
    commands: List[List[str]] = []
    emitted_coresets = set()
    emitted_buffers = set()
    for experiment in experiments:
        if "coreset" in stages:
            coreset_key = (
                experiment.artifact_log,
                experiment.dataset,
                experiment.method,
                experiment.seed,
                core_method,
                experiment.ratio,
            )
            if coreset_key not in emitted_coresets:
                commands.append(
                    [
                        python_executable,
                        "utils/coreset.py",
                        "--dname",
                        experiment.dataset,
                        "--method",
                        experiment.method,
                        "--device",
                        device,
                        "--save_log",
                        experiment.artifact_log,
                        "--core_method",
                        core_method,
                        "--reduction_rate",
                        experiment.ratio,
                        "--epochs",
                        str(coreset_epochs),
                        "--runs",
                        str(nruns),
                        "--seed",
                        str(experiment.seed),
                    ]
                )
                emitted_coresets.add(coreset_key)
        if "buffer" in stages:
            buffer_key = (
                experiment.artifact_log,
                experiment.dataset,
                experiment.method,
                experiment.seed,
                experiment.difficulty_type,
            )
            if buffer_key not in emitted_buffers:
                commands.append(
                    [
                        python_executable,
                        "buffer.py",
                        "--dname",
                        experiment.dataset,
                        "--method",
                        experiment.method,
                        "--device",
                        device,
                        "--save_log",
                        experiment.artifact_log,
                        "--teacher_epochs",
                        str(teacher_epochs),
                        "--num_experts",
                        str(num_experts),
                        "--traj_save_interval",
                        str(traj_save_interval),
                        "--param_save_interval",
                        str(param_save_interval),
                        "--difficulty_type",
                        experiment.difficulty_type,
                        "--seed_teacher",
                        str(experiment.seed),
                    ]
                )
                emitted_buffers.add(buffer_key)
        if "distill" in stages:
            commands.append(
                [
                    python_executable,
                    "distill.py",
                    "--dname",
                    experiment.dataset,
                    "--method",
                    experiment.method,
                    "--device",
                    device,
                    "--gpu_id",
                    str(gpu_id),
                    "--save_log",
                    experiment.save_log,
                    "--save_dir",
                    experiment.save_dir,
                    "--reduction_rate",
                    experiment.ratio,
                    "--seed",
                    str(experiment.seed),
                    "--coreset_seed",
                    str(experiment.seed),
                    "--coreset_init_path",
                    os.path.join(experiment.artifact_log, "Coreset"),
                    "--buffer_path",
                    os.path.join(experiment.artifact_log, "Buffer", f"{experiment.dataset}-buffer"),
                    "--core_method",
                    core_method,
                    "--ITER",
                    str(iterations),
                    "--eval_interval",
                    str(eval_interval),
                    "--syn_steps",
                    str(syn_steps),
                    "--expert_epochs",
                    str(expert_epochs),
                    "--min_start_epoch",
                    str(min_start_epoch),
                    "--max_start_epoch",
                    str(max_start_epoch),
                    "--max_start_epoch_s",
                    str(max_start_epoch_s),
                    "--nruns",
                    str(nruns),
                    "--test_model_iters",
                    str(test_model_iters),
                    "--expanding_window",
                    _bool_text(experiment.expanding_window),
                    "--beta",
                    str(experiment.beta),
                    "--soft_label",
                    _bool_text(experiment.soft_label),
                ]
            )
    return commands


def _parse_stages(stages: str) -> Tuple[str, ...]:
    parsed = tuple(stage.strip() for stage in stages.split(",") if stage.strip())
    allowed = {"coreset", "buffer", "distill"}
    unknown = set(parsed) - allowed
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    return parsed


def write_shell_script(commands: Iterable[Sequence[str]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        handle.write('export HGPRA_RUNTIME_HOME="${HGPRA_RUNTIME_HOME:-$PWD/.runtime_home}"\n')
        handle.write('export HOME="$HGPRA_RUNTIME_HOME"\n')
        handle.write('export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"\n')
        handle.write('mkdir -p "$HOME" "$MPLCONFIGDIR"\n\n')
        for command in commands:
            handle.write(shlex.join(command) + "\n")
    os.chmod(output_path, 0o755)


def find_missing_modules(python_executable: str, modules: Sequence[str]) -> List[str]:
    if not modules:
        return []
    probe = (
        "import importlib, sys\n"
        "failed=[]\n"
        "for module in sys.argv[1:]:\n"
        "    try:\n"
        "        importlib.import_module(module)\n"
        "    except Exception as exc:\n"
        "        failed.append(f'{module}: {type(exc).__name__}: {exc}')\n"
        "print('\\n'.join(failed))\n"
        "raise SystemExit(1 if failed else 0)\n"
    )
    result = subprocess.run(
        [python_executable, "-c", probe, *modules],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        return []
    failed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return failed or list(modules)


def format_dependency_error(python_executable: str, missing: Sequence[str], launcher_python: str) -> str:
    mismatch_note = ""
    if os.path.abspath(python_executable) != os.path.abspath(launcher_python):
        mismatch_note = (
            "\nNote: the runner was launched with a different Python executable:\n"
            f"  launcher:   {launcher_python}\n"
            f"  experiment: {python_executable}\n"
            "If your shell prompt shows the environment you intended to use, rerun with:\n"
            "  --python \"$(which python)\"\n"
            "or omit --python so the runner uses the active environment.\n"
        )
    return (
        "\nDependency preflight failed for Python executable:\n"
        f"  {python_executable}\n"
        f"{mismatch_note}\n"
        "Missing or broken import modules:\n"
        f"  {', '.join(missing)}\n\n"
        "Install the missing packages into the same environment. For lightweight packages:\n"
        f"  {shlex.quote(python_executable)} -m pip install numpy scipy scikit-learn tqdm matplotlib\n\n"
        "For PyTorch/PyG packages, install versions compatible with your CUDA and PyTorch build.\n"
        "Check the environment first with:\n"
        f"  {shlex.quote(python_executable)} -c \"import torch; print(torch.__version__, torch.version.cuda)\"\n\n"
        "You can also pass --python /path/to/env/bin/python to use another environment,\n"
        "or --skip-dependency-check if you intentionally want to bypass this check."
    )


def _parse_modules(value: str) -> Tuple[str, ...]:
    return tuple(module.strip() for module in value.split(",") if module.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["main", "ablation", "sensitivity", "efficiency"], default="ablation")
    parser.add_argument(
        "--grid",
        default="coauthor_cora:0.03",
        help="Semicolon grid, e.g. 'coauthor_cora:0.03,0.01;citeseer:0.03'",
    )
    parser.add_argument("--method", default="AllDeepSets")
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--log-root", default="logs_HGPRA")
    parser.add_argument("--save-dir", default="save_H")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--stages", default="coreset,buffer,distill")
    parser.add_argument("--coreset-epochs", type=int, default=200)
    parser.add_argument("--teacher-epochs", type=int, default=1000)
    parser.add_argument("--num-experts", type=int, default=10)
    parser.add_argument("--traj-save-interval", type=int, default=10)
    parser.add_argument("--param-save-interval", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--syn-steps", type=int, default=200)
    parser.add_argument("--expert-epochs", type=int, default=400)
    parser.add_argument("--min-start-epoch", type=int, default=5)
    parser.add_argument("--max-start-epoch", type=int, default=500)
    parser.add_argument("--max-start-epoch-s", type=int, default=10)
    parser.add_argument("--nruns", type=int, default=10)
    parser.add_argument("--test-model-iters", type=int, default=100)
    parser.add_argument("--core-method", choices=["herding", "kcenter", "random"], default="herding")
    parser.add_argument("--sensitivity-values", default="0,0.01,0.05,0.1,0.2")
    parser.add_argument(
        "--reuse-artifacts-within-dataset",
        action="store_true",
        help="Reuse coreset and expert buffers across variants that only change distillation options.",
    )
    parser.add_argument("--write-script", default="results/run_experiments.sh")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--check-modules",
        default=None,
        help="Comma-separated Python modules to verify before --execute. Defaults to modules required by the selected method.",
    )
    parser.add_argument("--skip-dependency-check", action="store_true")
    args = parser.parse_args()

    sensitivity_values = [float(value) for value in args.sensitivity_values.split(",") if value.strip()]
    stages = _parse_stages(args.stages)
    experiments = build_experiments(
        preset=args.preset,
        grid=args.grid,
        method=args.method,
        seed=args.seed,
        log_root=args.log_root,
        save_dir=args.save_dir,
        sensitivity_values=sensitivity_values,
        reuse_artifacts_within_dataset=args.reuse_artifacts_within_dataset,
    )
    commands = build_commands(
        experiments=experiments,
        python_executable=args.python,
        stages=stages,
        device=args.device,
        gpu_id=args.gpu_id,
        coreset_epochs=args.coreset_epochs,
        teacher_epochs=args.teacher_epochs,
        num_experts=args.num_experts,
        traj_save_interval=args.traj_save_interval,
        param_save_interval=args.param_save_interval,
        iterations=args.iterations,
        eval_interval=args.eval_interval,
        syn_steps=args.syn_steps,
        expert_epochs=args.expert_epochs,
        min_start_epoch=args.min_start_epoch,
        max_start_epoch=args.max_start_epoch,
        max_start_epoch_s=args.max_start_epoch_s,
        nruns=args.nruns,
        test_model_iters=args.test_model_iters,
        core_method=args.core_method,
    )
    write_shell_script(commands, args.write_script)
    for command in commands:
        print(shlex.join(command))
    print(f"Wrote {len(commands)} commands to {args.write_script}")

    if args.execute:
        if not args.skip_dependency_check:
            check_modules = (
                _parse_modules(args.check_modules)
                if args.check_modules
                else required_modules_for({experiment.method for experiment in experiments}, stages)
            )
            missing = find_missing_modules(args.python, check_modules)
            if missing:
                print(format_dependency_error(args.python, missing, sys.executable), file=sys.stderr)
                return 2
        code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        child_env = os.environ.copy()
        child_env.setdefault("KMP_INIT_AT_FORK", "FALSE")
        for command in commands:
            subprocess.run(command, cwd=code_dir, check=True, env=child_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
