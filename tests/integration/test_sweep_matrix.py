from __future__ import annotations

import numpy as np

from sweep.core import sweep, sweep_lite
from sweep.models import DEFAULT_SEED
from sweep.projection_core import orth_base
from tests.helpers.golden import load_golden_csv, load_golden_mtx


def test_sweep_matrix_all_modes_parity() -> None:
    sp_mat = load_golden_mtx("sweep_matrix_sparse_10x25.mtx")
    orth = orth_base(lin=25, col=5, seed=DEFAULT_SEED)
    golden_all = load_golden_csv("sweep_matrix_dense_10x25_all_modes.csv")

    # 1. count_none
    res_none = sweep(sp_mat, orthbase=orth, bin_mode=False, norm="none", verbose=False)
    np.testing.assert_allclose(res_none.proj, golden_all[:, 0:5], rtol=1e-12, atol=1e-12)

    # 2. bin_none
    res_bin = sweep(sp_mat, orthbase=orth, bin_mode=True, norm="none", verbose=False)
    np.testing.assert_allclose(res_bin.proj, golden_all[:, 5:10], rtol=1e-12, atol=1e-12)

    # 3. count_log
    res_log = sweep(sp_mat, orthbase=orth, bin_mode=False, norm="log", verbose=False)
    np.testing.assert_allclose(res_log.proj, golden_all[:, 10:15], rtol=1e-12, atol=1e-12)

    # 4. bin_logneg
    res_logneg = sweep(sp_mat, orthbase=orth, bin_mode=True, norm="logNeg", verbose=False)
    np.testing.assert_allclose(res_logneg.proj, golden_all[:, 15:20], rtol=1e-12, atol=1e-12)


def test_sweep_sparse_identity_matches_dense_for_neg_norms() -> None:
    sp_mat = load_golden_mtx("sweep_matrix_sparse_10x25.mtx")
    dense_mat = sp_mat.toarray()
    orth = orth_base(lin=25, col=5, seed=DEFAULT_SEED)

    # Neg counting
    p_sp_neg = sweep(sp_mat, orthbase=orth, bin_mode=False, norm="Neg", verbose=False).proj
    p_de_neg = sweep(dense_mat, orthbase=orth, bin_mode=False, norm="Neg", verbose=False).proj
    np.testing.assert_allclose(p_sp_neg, p_de_neg, rtol=1e-12, atol=1e-12)

    # logNeg counting
    p_sp_logneg = sweep(sp_mat, orthbase=orth, bin_mode=False, norm="logNeg", verbose=False).proj
    p_de_logneg = sweep(dense_mat, orthbase=orth, bin_mode=False, norm="logNeg", verbose=False).proj
    np.testing.assert_allclose(p_sp_logneg, p_de_logneg, rtol=1e-12, atol=1e-12)


def test_sweep_matrix_transpose_and_rna_seq_equivalence() -> None:
    mat = np.arange(20, dtype=np.float64).reshape(4, 5)
    orth = orth_base(lin=4, col=3, seed=DEFAULT_SEED)

    res_trans = sweep(mat, orthbase=orth, transpose=True, verbose=False).proj
    res_rna = sweep(mat, orthbase=orth, rna_seq_data=True, verbose=False).proj

    assert res_trans.shape == (5, 3)
    np.testing.assert_array_equal(res_trans, res_rna)


def test_sweep_lite_generic_matrix_matches_sweep() -> None:
    sp_mat = load_golden_mtx("sweep_matrix_sparse_10x25.mtx")
    orth = orth_base(lin=25, col=5, seed=DEFAULT_SEED)

    res_sweep = sweep(sp_mat, orthbase=orth, bin_mode=False, norm="none", verbose=False)
    res_lite = sweep_lite(sp_mat, psz=5, bin_mode=False, norm="none", verbose=False)

    np.testing.assert_allclose(res_sweep.proj, res_lite.proj, rtol=1e-12, atol=1e-12)
