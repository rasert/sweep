from __future__ import annotations

import numpy as np
from scipy.sparse import csr_array

from sweep.backend import (
    get_available_device,
    is_gpu_available,
    sparse_dense_matmul,
)
from sweep.models import DEFAULT_SEED
from sweep.projection_core import (
    PRIME_LIST_50,
    core_loop,
    lite_param,
    n_cores_def,
    orth_base,
)


def test_n_cores_def() -> None:
    assert n_cores_def(None) == 2
    assert n_cores_def(0) == 2
    assert n_cores_def(-4) == 2
    assert n_cores_def(8) == 8


def test_lite_param_generates_correct_dimensions() -> None:
    param = lite_param(psz=10, lenmax=400, seed=DEFAULT_SEED)
    assert len(param.pslist) == 50
    assert param.nps == 50
    assert param.mproj.shape == (50, 10)
    assert np.isclose(param.xnorm, np.sqrt(400.0 / 3.0))


def test_core_loop_matches_direct_orth_base_projection() -> None:
    # Build orth_base for 400 x 10
    orth = orth_base(lin=400, col=10, seed=DEFAULT_SEED)
    param = lite_param(psz=10, lenmax=400, seed=DEFAULT_SEED)

    # Synthetic non-zero indices (1-based for R parity in internal vectors)
    # E.g. indices 5, 23, 100 with counts 2.0, 1.0, 5.0
    hdv_idx = np.array([5, 23, 100], dtype=np.int64)
    hdv_count = np.array([2.0, 1.0, 5.0], dtype=np.float64)

    # 1. Counting mode
    res_core = core_loop(
        param.xnorm,
        10,
        param.mproj,
        PRIME_LIST_50,
        50,
        hdv_idx,
        hdv_count,
        nk=1000,
        bin_mode=False,
        norm="none",
    )

    # Direct multiplication with full matrix (0-based indices in Python: 4, 22, 99)
    full_row = np.zeros(400, dtype=np.float64)
    full_row[hdv_idx - 1] = hdv_count
    expected = full_row @ orth.mat

    np.testing.assert_allclose(res_core, expected, rtol=1e-12, atol=1e-12)

    # 2. Binary mode
    res_core_bin = core_loop(
        param.xnorm,
        10,
        param.mproj,
        PRIME_LIST_50,
        50,
        hdv_idx,
        None,
        nk=1000,
        bin_mode=True,
        norm="none",
    )
    full_row_bin = np.zeros(400, dtype=np.float64)
    full_row_bin[hdv_idx - 1] = 1.0
    expected_bin = full_row_bin @ orth.mat

    np.testing.assert_allclose(res_core_bin, expected_bin, rtol=1e-12, atol=1e-12)


def test_backend_device_and_sparse_matmul() -> None:
    device = get_available_device("auto")
    if is_gpu_available():
        assert device == "cuda"
    else:
        assert device == "cpu"

    # Verify sparse dense matmul on both CPU and GPU gives identical results
    sp_mat = csr_array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float64)
    dense_mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    cpu_res = sparse_dense_matmul(sp_mat, dense_mat, device="cpu")
    expected = sp_mat.toarray() @ dense_mat
    np.testing.assert_allclose(cpu_res, expected, rtol=1e-14)

    if is_gpu_available():
        gpu_res = sparse_dense_matmul(sp_mat, dense_mat, device="cuda")
        np.testing.assert_allclose(gpu_res, expected, rtol=1e-14)
