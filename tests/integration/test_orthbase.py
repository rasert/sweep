from __future__ import annotations

import numpy as np

from sweep.models import DEFAULT_SEED, DEFAULT_VERSION, OrthBaseResult
from sweep.projection_core import orth_base
from tests.helpers.golden import load_golden_csv


def test_orth_base_aa_matches_golden_file() -> None:
    # 1. Default float32 mode (lightweight, matches golden file within FP32 precision)
    res_f32 = orth_base(mask=[1, 0, 1], col=10, seqtype="AA", seed=DEFAULT_SEED)
    assert isinstance(res_f32, OrthBaseResult)
    assert res_f32.seed == DEFAULT_SEED
    assert res_f32.version == DEFAULT_VERSION
    assert res_f32.mat.shape == (400, 10)
    assert res_f32.mat.dtype == np.float32

    golden = load_golden_csv("orthbase_aa_400x10.csv")
    np.testing.assert_allclose(res_f32.mat, golden, rtol=1e-6, atol=1e-6)

    # 2. Strict float64 mode (exact 1e-12 bitwise parity with R)
    res_f64 = orth_base(mask=[1, 0, 1], col=10, seqtype="AA", seed=DEFAULT_SEED, dtype=np.float64)
    assert res_f64.mat.dtype == np.float64
    np.testing.assert_allclose(res_f64.mat, golden, rtol=1e-12, atol=1e-12)


def test_orth_base_nt_matches_golden_file() -> None:
    res = orth_base(mask=[2, 1, 2], col=10, seqtype="NT", seed=DEFAULT_SEED, dtype=np.float64)

    assert isinstance(res, OrthBaseResult)
    assert res.mat.shape == (256, 10)
    assert res.mat.dtype == np.float64

    golden = load_golden_csv("orthbase_nt_256x10.csv")
    np.testing.assert_allclose(res.mat, golden, rtol=1e-12, atol=1e-12)


def test_orth_base_with_explicit_lin() -> None:
    res = orth_base(lin=400, col=10, seed=DEFAULT_SEED, dtype=np.float64)
    assert res.mat.shape == (400, 10)
    assert res.mat.dtype == np.float64

    golden = load_golden_csv("orthbase_aa_400x10.csv")
    np.testing.assert_allclose(res.mat, golden, rtol=1e-12, atol=1e-12)
