import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from model.preprocessing import generate_G_from_H


class PreprocessingNumpyCompatibilityTests(unittest.TestCase):
    def test_generate_g_from_h_uses_numpy2_compatible_matrix_conversion(self):
        data = SimpleNamespace(
            edge_index=np.array(
                [
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ]
            )
        )

        out = generate_G_from_H(data)

        self.assertIsInstance(out.edge_index, torch.Tensor)
        self.assertEqual(tuple(out.edge_index.shape), (3, 3))
        self.assertTrue(torch.isfinite(out.edge_index).all())


if __name__ == "__main__":
    unittest.main()
