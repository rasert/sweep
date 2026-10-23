from __future__ import annotations

import numpy as np
from scipy.sparse import csr_array

from sweep.hdv import make_log, read_mask, read_mask_vec
from sweep.indexing import seq2num


def test_read_mask_and_read_mask_vec_consistency() -> None:
    seq_str = "ACDACDACD"
    seq_num = seq2num(seq_str, "AA")
    mask = [1, 0, 1]
    lenmax = 400

    # 1. bin=False, norm='none'
    r_vec_count = read_mask_vec(seq_num, mask, "AA", bin_mode=False, norm="none", lenmax=lenmax)
    r_list_count = read_mask(seq_num, mask, "AA", bin_mode=False, norm="none", lenmax=lenmax)
    nz_idx = np.where(r_vec_count != 0)[0]
    np.testing.assert_array_equal(nz_idx, r_list_count["idx"])
    np.testing.assert_array_equal(r_vec_count[nz_idx], r_list_count["count"])

    # 2. bin=True, norm='none'
    r_vec_bin = read_mask_vec(seq_num, mask, "AA", bin_mode=True, norm="none", lenmax=lenmax)
    r_list_bin = read_mask(seq_num, mask, "AA", bin_mode=True, norm="none", lenmax=lenmax)
    nz_idx_bin = np.where(r_vec_bin != 0)[0]
    np.testing.assert_array_equal(nz_idx_bin, r_list_bin["idx"])
    assert np.all(r_vec_bin[nz_idx_bin] == 1.0)
    assert r_list_bin["count"] is None

    # 3. bin=False, norm='log'
    r_vec_log = read_mask_vec(seq_num, mask, "AA", bin_mode=False, norm="log", lenmax=lenmax)
    r_list_log = read_mask(seq_num, mask, "AA", bin_mode=False, norm="log", lenmax=lenmax)
    nz_idx_log = np.where(r_vec_log != 0)[0]
    np.testing.assert_array_equal(nz_idx_log, r_list_log["idx"])
    np.testing.assert_allclose(r_vec_log[nz_idx_log], r_list_log["count"], rtol=1e-12)

    # 4. bin=False, norm='Neg'
    r_vec_neg = read_mask_vec(seq_num, mask, "AA", bin_mode=False, norm="Neg", lenmax=lenmax)
    r_list_neg = read_mask(seq_num, mask, "AA", bin_mode=False, norm="Neg", lenmax=lenmax)
    np.testing.assert_allclose(r_vec_neg, r_list_neg["count"], rtol=1e-12)
    assert np.all(r_vec_neg[r_vec_count == 0] == -1.0)

    # 5. bin=False, norm='logNeg'
    r_vec_logneg = read_mask_vec(seq_num, mask, "AA", bin_mode=False, norm="logNeg", lenmax=lenmax)
    r_list_logneg = read_mask(seq_num, mask, "AA", bin_mode=False, norm="logNeg", lenmax=lenmax)
    np.testing.assert_allclose(r_vec_logneg, r_list_logneg["count"], rtol=1e-12)
    assert np.all(r_vec_logneg[r_vec_count == 0] == -1.0)


def test_make_log_transforms_dense_matrices_for_all_modes() -> None:
    mat = np.array([[0.0, 1.0], [10.0, 100.0]], dtype=np.float64)

    # bin=False, norm='none'
    np.testing.assert_array_equal(make_log(mat, bin_mode=False, norm="none"), mat)

    # bin=True, norm='none'
    np.testing.assert_array_equal(
        make_log(mat, bin_mode=True, norm="none"),
        np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float64),
    )

    # bin=False, norm='log'
    np.testing.assert_allclose(
        make_log(mat, bin_mode=False, norm="log"),
        np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64),
    )

    # bin=False, norm='Neg'
    np.testing.assert_array_equal(
        make_log(mat, bin_mode=False, norm="Neg"),
        np.array([[-1.0, 1.0], [10.0, 100.0]], dtype=np.float64),
    )

    # bin=False, norm='logNeg'
    np.testing.assert_allclose(
        make_log(mat, bin_mode=False, norm="logNeg"),
        np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float64),
    )

    # bin=True, norm='logNeg'
    np.testing.assert_array_equal(
        make_log(mat, bin_mode=True, norm="logNeg"),
        np.array([[-1.0, 1.0], [1.0, 1.0]], dtype=np.float64),
    )


def test_make_log_sparse_matches_dense_truth_table() -> None:
    dense_mat = np.array([[0.0, 1.0], [10.0, 100.0]], dtype=np.float64)
    sp_mat = csr_array(dense_mat)

    # In modes none and log, make_log on sparse preserves sparse CSR representation
    sp_none = make_log(sp_mat, bin_mode=False, norm="none")
    assert isinstance(sp_none, csr_array)
    np.testing.assert_array_equal(sp_none.toarray(), dense_mat)

    sp_bin = make_log(sp_mat, bin_mode=True, norm="none")
    assert isinstance(sp_bin, csr_array)
    np.testing.assert_array_equal(sp_bin.toarray(), np.array([[0.0, 1.0], [1.0, 1.0]]))

    sp_log = make_log(sp_mat, bin_mode=False, norm="log")
    assert isinstance(sp_log, csr_array)
    np.testing.assert_allclose(sp_log.toarray(), np.array([[0.0, 0.0], [1.0, 2.0]]))
