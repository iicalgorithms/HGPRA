import os
import sys
import unittest


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from pretreatment.convert_datasets_to_pygDataset import _load_raw_dataset_functions
from buffer import snapshot_parameters
from torch_geometric.data import Data
from utils.dataloader import normalize_labels


class DatasetLoadingTests(unittest.TestCase):
    def test_raw_dataset_loader_does_not_require_ipdb(self):
        loaders = _load_raw_dataset_functions()

        self.assertIn("load_citation_dataset", loaders)
        self.assertIn("load_LE_dataset", loaders)

    def test_buffer_snapshots_do_not_alias_live_parameters(self):
        import torch

        layer = torch.nn.Linear(2, 1, bias=False)
        snapshot = snapshot_parameters(layer)
        before = snapshot[0].clone()

        with torch.no_grad():
            layer.weight.add_(1.0)

        self.assertTrue(torch.equal(snapshot[0], before))

    def test_labels_are_normalized_to_zero_based_contiguous_ids(self):
        import torch

        data = Data(y=torch.tensor([1, 3, 3, 1, 2]))

        num_classes = normalize_labels(data)

        self.assertEqual(num_classes, 3)
        self.assertEqual(data.y.tolist(), [0, 2, 2, 0, 1])


if __name__ == "__main__":
    unittest.main()
