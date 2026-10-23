from __future__ import annotations

import numpy as np

from sweep.phylo import costquebra


def test_costquebra_homogeneous_distribution() -> None:
    # All 1s adjacent -> cost 1.0
    z = np.array([1, 1, 1, 1], dtype=np.int32)
    assert costquebra(z) == 1.0


def test_costquebra_single_occurrence() -> None:
    z = np.array([0, 1, 0], dtype=np.int32)
    assert costquebra(z) == 1.0


def test_costquebra_fragmented_taxa() -> None:
    # Two ones separated by a zero: [1, 0, 1] -> N=2, adjacent pairs=0 -> (1+0)/2 = 0.5
    z = np.array([1, 0, 1], dtype=np.int32)
    assert costquebra(z) == 0.5

    # [1, 1, 0, 1] -> N=3, diff == 1 has one adjacent pair -> (1 + 1) / 3 = 2/3
    z3 = np.array([1, 1, 0, 1], dtype=np.int32)
    np.testing.assert_allclose(costquebra(z3), 2.0 / 3.0)
