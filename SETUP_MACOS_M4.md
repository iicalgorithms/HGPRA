# macOS M4 Environment Setup

This project is easiest to run on Apple Silicon by keeping the Python environment
small and avoiding optional CUDA-oriented packages unless a selected backbone
actually needs them.

## Recommended Strategy

Use the existing `GNN` conda environment if it already imports `torch`,
`torch_geometric`, and `torch_scatter`. These are the hardest packages to align
on macOS. The code has been adjusted so that `wandb`, `deeprobust`, and
`tensorboardX` are no longer required for the default HGPRA pipeline.

Check the current environment:

```bash
conda activate GNN
cd code

python - <<'PY'
import importlib
for name in ["torch", "torch_geometric", "torch_scatter", "numpy", "scipy", "sklearn", "tqdm"]:
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "ok"))
PY
```

Run the project tests:

```bash
python -m unittest discover -s tests -v
```

## Fresh Environment

If you prefer a clean environment, start with Python 3.10 and install PyTorch
with pip. PyTorch's macOS binaries are pip-first, and PyG provides wheel URLs
that are keyed by the installed PyTorch version.

```bash
conda create -n hgpra-m4 python=3.10 -y
conda activate hgpra-m4

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio
python -m pip install numpy scipy scikit-learn tqdm matplotlib pandas
python -m pip install torch_geometric

TORCH_VERSION=$(python - <<'PY'
import torch
print(torch.__version__.split("+")[0])
PY
)
python -m pip install pyg_lib torch_scatter -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html"
```

Install optional packages only when needed:

```bash
# Required only for UniGCNII.
python -m pip install torch_sparse -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html"

# Required only for the DHG-based HyperGCN wrapper.
python -m pip install dhg

# Optional logging.
python -m pip install tensorboard wandb
```

## Device Choice on M4

Use CPU first:

```bash
python scripts/run_experiment_grid.py \
  --preset ablation \
  --grid "coauthor_cora:0.03" \
  --device cpu \
  --gpu-id -1 \
  --write-script results/run_ablation_cpu.sh
```

The code intentionally does not auto-select MPS. The current pipeline uses
sparse reductions and PyG-style indexing, and those paths are more predictable
on CPU. After CPU runs work, you can test MPS manually by changing the training
entrypoints to `--device mps`; treat that as experimental.

## Data Layout

The first runtime failure after dependency setup is usually missing data. The
loader expects:

```text
code/data/
├── AllSet_all_raw_data/
│   └── cocitation/
└── pyg_data/
    └── hypergraph_dataset_updated/
```

For `coauthor_cora` (reported as Cora-CA in the paper), place files under:

```text
code/data/AllSet_all_raw_data/coauthorship/cora/
```

For `citeseer` and `pubmed`, place the raw cocitation files under:

```text
code/data/AllSet_all_raw_data/cocitation/
```

## Smoke Checks

Dependency and import smoke checks:

```bash
python -m unittest discover -s tests -v

python scripts/run_experiment_grid.py \
  --preset ablation \
  --grid "coauthor_cora:0.03" \
  --stages buffer \
  --device cpu \
  --gpu-id -1 \
  --teacher-epochs 12 \
  --num-experts 1 \
  --traj-save-interval 1 \
  --param-save-interval 1 \
  --write-script /tmp/hgpra-smoke.sh
```

Add `--execute` only after the required dataset files are in place.
