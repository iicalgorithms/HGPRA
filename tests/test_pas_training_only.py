import os
import sys
import unittest

import torch


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from utils.utils import node_difficulty_bipartite


class PaSTrainingOnlyTests(unittest.TestCase):
    def test_node_difficulty_ignores_non_training_labels(self):
        edge_index = torch.tensor(
            [
                [0, 2, 1, 3],
                [0, 0, 1, 1],
            ],
            dtype=torch.long,
        )
        train_idx = torch.tensor([0, 1], dtype=torch.long)

        labels_a = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        labels_b = torch.tensor([0, 1, 1, 0], dtype=torch.long)

        scores_a = node_difficulty_bipartite(
            edge_index=edge_index,
            labels=labels_a,
            num_nodes=4,
            train_idx=train_idx,
            device="cpu",
        )
        scores_b = node_difficulty_bipartite(
            edge_index=edge_index,
            labels=labels_b,
            num_nodes=4,
            train_idx=train_idx,
            device="cpu",
        )

        self.assertTrue(torch.allclose(scores_a, scores_b, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
