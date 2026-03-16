# HGPRA Codebase

This directory contains the source code for the paper **HGPRA: Perception-Aware Hypergraph Condensation via Progressive Representation Anchoring**.

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

This repository does not currently include a pinned `requirements.txt`, so you will need to install the Python dependencies manually.

A practical starting point is:

- Python 3.8+
- PyTorch
- PyTorch Geometric
- `torch_sparse`
- `torch_scatter`
- `numpy`
- `scipy`
- `scikit-learn`
- `tensorboardX`
- `deeprobust`
- `dhg`
- `tqdm`

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

For citation datasets, `code/utils/dataloader.py` expects:

- `./data/AllSet_all_raw_data/cocitation/` for `cora`, `citeseer`, `pubmed`
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

If you use this codebase, please cite the corresponding HGPRA paper.

## Acknowledgment

Parts of the data loading / HyperGNN support pipeline build on common hypergraph learning tooling and PyG-style dataset processing.
