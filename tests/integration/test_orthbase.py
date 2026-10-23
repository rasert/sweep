from __future__ import annotations

import numpy as np

from sweep.models import DEFAULT_SEED, DEFAULT_VERSION, OrthBaseResult
from sweep.projection_core import orth_base
from tests.helpers.golden import load_golden_csv


def test_orth_base_aa_matches_golden_file() -> None:
    res = orth_base(mask=[1, 0, 1], col=10, seqtype="AA", seed=DEFAULT_SEED)

    assert isinstance(res, OrthBaseResult)
    assert res.seed == DEFAULT_SEED
    assert res.version == DEFAULT_VERSION
    assert res.mat.shape == (400, 10)

    golden = load_golden_csv("orthbase_aa_400x10.csv")
    np.testing.assert_allclose(res.mat, golden, rtol=1e-12, atol=1e-12)


def test_orth_base_nt_matches_golden_file() -> None:
    res = orth_base(mask=[2, 1, 2], col=10, seqtype="NT", seed=DEFAULT_SEED)

    assert isinstance(res, OrthBaseResult)
    assert res.mat.shape == (256, 10)

    golden = load_golden_csv("orthbase_nt_256x10.csv")
    np.testing.assert_allclose(res.mat, golden, rtol=1e-12, atol=1e-12)


def test_orth_base_with_explicit_lin() -> None:
    res = orth_base(lin=400, col=10, seed=DEFAULT_SEED)
    assert res.mat.shape == (400, 10)

    golden = load_golden_csv("orthbase_aa_400x10.csv")
    np.testing.assert_allclose(res.mat, golden, rtol=1e-12, atol=1e-12)
