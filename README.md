# HGPRA Codebase

This directory contains the source code for the paper **HGPRA: Training-Dynamics-Aware Hypergraph Condensation via Progressive Representation Anchoring**.

The implementation follows a two-stage pipeline:

1. Train expert HyperGNN models on the original hypergraph and save their parameter trajectories.
2. Distill a compact structure-free surrogate hypergraph from those trajectories and evaluate it with downstream HyperGNN backbones.

## Overview

HGPRA targets supervised hypergraph node classification. The code supports multiple HyperGNN backbones and uses a trajectory-based condensation workflow:

- `code/buffer.py`: trains expert models and stores replay buffers / trajectories.
- `code/distill.py`: performs condensation using the saved trajectories.
- `code/meta_gradient.py`: core implementation of the HGPRA optimization procedure.
- `code/utils/dataloader.py`: dataset loading and model-specific preprocessing.
- `code/model/`: backbone models and wrappers.
- `code/pretreatment/`: raw-data conversion and dataset preparation utilities.

## Directory Structure

```text
code/
├── README.md
├── buffer.py
├── distill.py
├── meta_gradient.py
├── model_data_parser.py
├── model/
│   ├── model_loader.py
│   ├── hgnn.py
│   ├── hgnn_plus.py
│   ├── hnhn.py
│   ├── hypergcn.py
│   ├── hypergcn_wrapper.py
│   ├── setgnn_wrapper.py
│   ├── unigat.py
│   └── ...
├── pretreatment/
│   ├── convert_datasets_to_pygDataset.py
│   └── load_other_datasets.py
└── utils/
    ├── dataloader.py
    ├── train.py
    ├── inference.py
    ├── coreset.py
    └── ...
```

## Supported Datasets

The dataloader currently recognizes the following dataset names:

- `cora`
- `citeseer`
- `pubmed`
- `20newsW100`
- `ModelNet40`
- `zoo`
- `NTU2012`
- `Mushroom`
- `coauthor_cora`
- `coauthor_dblp`
- `yelp`
- `amazon-reviews`
- `walmart-trips`
- `house-committees`
- `walmart-trips-100`
- `house-committees-100`
- `tencent_2k`

Raw datasets are expected under `./data/AllSet_all_raw_data/` and processed PyG-style datasets are stored under `./data/pyg_data/hypergraph_dataset_updated/`.

## Supported Backbones

The current model loader supports the following methods:

- `AllDeepSets`
- `AllSetTransformer`
- `CEGCN`
- `CEGAT`
- `HyperGCN`
- `HGNN`
- `HNHN`
- `HCHA`
- `MLP`
- `UniGCNII`
- `UniGAT`

In the paper, `AllDeepSets` is used as the default backbone in many experiments, while cross-architecture evaluation also includes `HyperGCN`, `HGNN`, and `AllSetTransformer`.

## Environment

For macOS on Apple Silicon, see `SETUP_MACOS_M4.md` first. For the paper-level
experiment grid, table-generation workflow, and result provenance rules, see
`EXPERIMENTS_ICDM2026.md`.

The recommended path on M4 is to use Python 3.10, install PyTorch/PyG with pip,
and keep optional backends such as `dhg` and `torch_sparse` out of the default
AllDeepSets pipeline.

For the local Apple Silicon reproduction path, `requirements-macos-m4.txt`
records the minimal package set used by the default CPU-only pipeline. It is not
a fully pinned cross-platform lock file, because PyTorch/PyG sparse wheels are
platform-specific.

A practical starting point is:

- Python 3.10
- PyTorch
- PyTorch Geometric
- `torch_scatter`
- `numpy`
- `scipy`
- `scikit-learn`
- `tqdm`
- `matplotlib` for result figures
- `pandas` for convenient result inspection

Optional packages:

- `torch_sparse` for UniGCNII.
- `dhg` for the DHG-based HyperGCN wrapper.
- `tensorboard` or `tensorboardX` for TensorBoard logging.
- `wandb` for Weights \& Biases logging.

If you are using CUDA, make sure your PyTorch / PyG / sparse operator versions are mutually compatible.

## Data Preparation

The code assumes that the processed hypergraph datasets either already exist or can be generated from the raw files.

Useful scripts:

- `code/pretreatment/convert_datasets_to_pygDataset.py`
- `code/pretreatment/load_other_datasets.py`

Typical expected layout:

```text
data/
├── AllSet_all_raw_data/
│   ├── cocitation/
│   ├── coauthorship/
│   └── ...
└── pyg_data/
    └── hypergraph_dataset_updated/
```

For the paper datasets, `code/utils/dataloader.py` expects:

- `./data/AllSet_all_raw_data/coauthorship/cora/` for `coauthor_cora` (reported as Cora-CA in the paper)
- `./data/AllSet_all_raw_data/cocitation/` for `citeseer`, `pubmed`
- `./data/AllSet_all_raw_data/coauthorship/` for `coauthor_cora`, `coauthor_dblp`

## Stage 1: Train Expert Trajectories

Before running HGPRA condensation, you need expert trajectory buffers.

Example:

```bash
cd code
python buffer.py \
  --dname cora \
  --method AllDeepSets \
  --device cuda:0 \
  --teacher_epochs 1000 \
  --num_experts 10 \
  --traj_save_interval 10 \
  --difficulty_type node \
  --save_log logs_HGPRA
```

This script will:

- load the full hypergraph,
- train expert models,
- optionally apply curriculum-style training order via `difficulty_type`,
- save replay buffers under a directory like:

```text
logs_HGPRA/Buffer/cora-buffer/
```

Important arguments in `buffer.py`:

- `--dname`: dataset name
- `--method`: backbone model
- `--device`: e.g. `cuda:0` or `cpu`
- `--teacher_epochs`: expert training epochs
- `--num_experts`: number of expert trajectories
- `--difficulty_type`: `random`, `node`, or `hyperedge`
- `--lam`, `--T`, `--scheduler`: curriculum scheduling parameters

## Stage 2: Run HGPRA Condensation

After expert trajectories are available, run:

```bash
cd code
python distill.py \
  --dname cora \
  --method AllDeepSets \
  --gpu_id 0 \
  --reduction_rate 0.03 \
  --save_log logs_HGPRA \
  --save_dir save_H \
  --expert_epochs 400 \
  --ITER 100 \
  --eval_interval 1 \
  --expanding_window True
```

This script will:

- load the processed dataset,
- read expert trajectory buffers from `logs_HGPRA/Buffer/<dataset>-buffer/`,
- initialize a condensed surrogate,
- optimize synthetic features / labels,
- periodically evaluate downstream performance.

Important arguments in `distill.py`:

- `--dname`: dataset name
- `--method`: backbone used for condensation / evaluation
- `--gpu_id`: GPU id
- `--reduction_rate`: condensation ratio
- `--save_log`: root log directory, must match the buffer stage if you want to reuse saved trajectories
- `--save_dir`: directory for saving condensed artifacts
- `--expert_epochs`: expert training horizon used in matching
- `--ITER`: outer optimization iterations
- `--syn_steps`: student training steps on synthetic data
- `--expanding_window`: enable the progressive window matching strategy
- `--beta`: coefficient for the auxiliary matching term
- `--soft_label`: whether to use soft label learning

## Minimal Reproduction Workflow

For a typical run on `cora` with `AllDeepSets`:

```bash
cd code

python buffer.py \
  --dname cora \
  --method AllDeepSets \
  --device cuda:0 \
  --save_log logs_HGPRA \
  --difficulty_type node

python distill.py \
  --dname cora \
  --method AllDeepSets \
  --gpu_id 0 \
  --save_log logs_HGPRA \
  --save_dir save_H \
  --reduction_rate 0.03 \
  --expanding_window True
```

## Outputs

Typical outputs include:

- expert trajectory buffers:
  - `logs_*/Buffer/<dataset>-buffer/replay_buffer_*.pt`
- distillation logs:
  - `logs_*/Distill/<dataset>-reduce_<ratio>-<timestamp>/train.log`
- TensorBoard summaries
- saved condensed data / synthetic artifacts under `save_H/`

## Reproducible Experiment Scripts

The `scripts/` directory provides a thin orchestration layer around the existing
entrypoints. It does not change the training algorithm; it generates repeatable
commands, collects log files, and renders tables or figures from the resulting
CSV files.

Dry-run an ablation grid and write a runnable shell script:

```bash
cd code
python scripts/run_experiment_grid.py \
  --preset ablation \
  --grid "coauthor_cora:0.03" \
  --device cuda:0 \
  --gpu-id 0 \
  --write-script results/run_ablation.sh
```

Execute the generated commands directly:

```bash
python scripts/run_experiment_grid.py \
  --preset ablation \
  --grid "coauthor_cora:0.03" \
  --device cuda:0 \
  --gpu-id 0 \
  --execute
```

When `--execute` is used, the runner first checks whether the selected Python
environment can import the dependencies needed by the selected backbone. The
default `AllDeepSets`/`AllSetTransformer` path does not require `dhg` or
`torch_sparse`; those optional backends are checked only for methods that need
them, such as `HyperGCN` or `UniGCNII`. If the server has more than one conda
environment, point the runner to the intended interpreter:

```bash
python scripts/run_experiment_grid.py \
  --preset ablation \
  --grid "coauthor_cora:0.03" \
  --python /home/liuyang/anaconda3/envs/TAG/bin/python \
  --device cuda:0 \
  --gpu-id 0 \
  --execute
```

If the preflight reports missing modules, install them into the same interpreter,
for example:

```bash
/home/liuyang/anaconda3/envs/TAG/bin/python -m pip install numpy scipy scikit-learn tqdm tensorboardX
```

Install PyTorch and PyG packages (`torch`, `torch_geometric`, `torch_sparse`,
`torch_scatter`) with versions matching your CUDA/PyTorch setup when you run
backbones that require them. Use `--skip-dependency-check` only after you have
verified the environment manually.

If execution reaches a message such as
`path to raw hypergraph dataset "./data/AllSet_all_raw_data/cocitation/" does not exist`,
the Python environment is ready for the selected default backbone but the data
directory is missing. Place the raw and/or processed datasets under the layout
described above, or run from a workspace where `code/data/` already contains the
expected files.

Useful presets:

- `main`: HGPRA runs for the selected dataset-ratio grid.
- `ablation`: HGPRA, w/o PaS, w/o PRA, and w/o KGED.
- `sensitivity`: distillation-weight sensitivity using `--sensitivity-values`.
- `efficiency`: HGPRA runs intended for runtime collection.

After experiments finish, collect logs and render outputs:

```bash
python scripts/collect_results.py \
  --log-root logs_HGPRA \
  --out results/summary.csv \
  --latest-per-config

python scripts/make_tables.py \
  --input results/summary.csv \
  --out results/generated_table.tex \
  --caption "Generated HGPRA experiment summary." \
  --label "tab:generated_results"

python scripts/make_figures.py \
  --input results/summary.csv \
  --out-dir results/figures \
  --x-field beta \
  --y-field test_acc_percent \
  --group-field dataset
```

To regenerate the tables used directly by the manuscript from the structured
paper CSV files:

```bash
python scripts/write_paper_tables.py \
  --results-dir results/paper \
  --out-dir ../tables
```

The generated fragments are included from `../main.tex`:

- `../tables/performance_comparison.tex`
- `../tables/hypergnn_comparison.tex`
- `../tables/ablation.tex`
- `../tables/time.tex`

The CSV files under `results/paper/` currently mirror the values already present
in the manuscript. Replace those values only with completed experiment outputs
or documented external baseline results.

## Paper Result Tables and Commands

The following table maps manuscript outputs to the commands used to regenerate
the runnable HGPRA grids and then rebuild the LaTeX table fragments. Baseline
rows that are not implemented by this repository must be imported only from
documented external result files.

| Manuscript output | Result source | Commands |
| --- | --- | --- |
| Main accuracy, Table `performance_comparison` | `results/paper/main_performance.csv` | `python scripts/run_experiment_grid.py --preset main --grid "coauthor_cora:0.03,0.01,0.005;citeseer:0.03,0.01,0.005;pubmed:0.001,0.0007,0.0005;20newsW100:0.01,0.005,0.003;ModelNet40:0.01,0.005,0.003" --method AllDeepSets --device cpu --gpu-id -1 --write-script results/run_main_alldeepsets.sh` then `./results/run_main_alldeepsets.sh` |
| Cross-architecture, Table `hypergnn_comparison` | `results/paper/cross_architecture.csv` | Generate one script per backbone with `python scripts/run_experiment_grid.py --preset main --grid "coauthor_cora:0.03;citeseer:0.03;pubmed:0.001;20newsW100:0.01;ModelNet40:0.01" --method <BACKBONE> --device cpu --gpu-id -1 --write-script results/run_transfer_<BACKBONE>.sh`, where `<BACKBONE>` is `AllDeepSets`, `HGNN`, `HyperGCN`, or `AllSetTransformer`; then run the generated scripts. |
| Component ablation, Table `ablation` | `results/paper/component_ablation.csv` | `python scripts/run_experiment_grid.py --preset ablation --grid "coauthor_cora:0.03;citeseer:0.03" --method AllDeepSets --device cpu --gpu-id -1 --write-script results/run_ablation_pas_pra_kged.sh` then `./results/run_ablation_pas_pra_kged.sh` |
| KGED sensitivity, Fig. `hyperparameter` | `results/paper/sensitivity_beta.csv` | `python scripts/run_experiment_grid.py --preset sensitivity --grid "20newsW100:0.01;ModelNet40:0.01" --method AllDeepSets --sensitivity-values "0,0.01,0.04,0.1,0.2" --device cpu --gpu-id -1 --write-script results/run_sensitivity_kged.sh` then `./results/run_sensitivity_kged.sh` |
| Runtime, Table `time` | `results/paper/runtime.csv` | `python scripts/run_experiment_grid.py --preset efficiency --grid "coauthor_cora:0.03,0.01,0.005;20newsW100:0.01,0.005,0.003" --method AllDeepSets --device cpu --gpu-id -1 --write-script results/run_efficiency.sh` then `./results/run_efficiency.sh` |

After updating the CSV files, regenerate manuscript tables with:

```bash
python scripts/write_paper_tables.py \
  --results-dir results/paper \
  --out-dir ../tables
```

For a fast smoke test, reduce the expensive training arguments:

```bash
python scripts/run_experiment_grid.py \
  --preset ablation \
  --grid "coauthor_cora:0.03" \
  --log-root logs_HGPRA/smoke_snapshot_window \
  --coreset-epochs 1 \
  --teacher-epochs 12 \
  --num-experts 1 \
  --traj-save-interval 1 \
  --param-save-interval 1 \
  --iterations 1 \
  --syn-steps 1 \
  --expert-epochs 10 \
  --min-start-epoch 1 \
  --max-start-epoch 10 \
  --max-start-epoch-s 2 \
  --nruns 1 \
  --test-model-iters 1
```

## Notes on Model-Specific Preprocessing

Different backbones use different preprocessing paths in `code/utils/dataloader.py`:

- `HGNN` / `HCHA`: construct propagation matrix `G`
- `HyperGCN`: keeps a V2E edge index and applies dedicated preprocessing
- `AllDeepSets` / `AllSetTransformer`: use the SetGNN-style processing path
- `UniGCNII` / `UniGAT`: require additional degree normalization terms

Because of this, if you switch `--method`, the same dataset may be converted slightly differently before training.

## Common Pitfalls

- Make sure the buffer stage and distillation stage use the same `--save_log`; otherwise `distill.py` will not find the expert replay buffers.
- Make sure the dataset name matches the values hard-coded in `code/utils/dataloader.py`.
- Some datasets require raw files in specific subdirectories, especially citation and coauthorship datasets.
- PyTorch Geometric and sparse extensions must match your PyTorch/CUDA version.
- Several scripts assume CUDA by default. If running on CPU, explicitly set the device-related arguments.

## Citation

If you use this codebase, please cite the corresponding HGPRA paper after the
anonymous review period.
