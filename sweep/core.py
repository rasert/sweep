from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array, sparray, spmatrix

from sweep.backend import (
    dense_dense_matmul,
    get_available_device,
    sparse_dense_matmul,
)
from sweep.checks import sw_check_orthbase, sw_checks
from sweep.hdv import extract_hdv, make_log
from sweep.indexing import def_mask
from sweep.models import (
    DEFAULT_VERSION,
    DeviceType,
    DTypeLike,
    NormMode,
    OrthBaseResult,
    SeqType,
    SweepInfo,
    SweepResult,
)
from sweep.projection_core import (
    n_cores_def,
    orth_base,
)


def _project_sparse_matrix(
    sp_mat: csr_array,
    orth_mat: NDArray[np.floating[Any]],
    bin_mode: bool,
    norm: NormMode,
    device: DeviceType,
) -> NDArray[np.floating[Any]]:
    """Applies projection to sparse CSR matrix without array densification."""
    work_dtype = orth_mat.dtype
    if norm == "none":
        if bin_mode:
            data = np.ones_like(sp_mat.data, dtype=work_dtype)
            sp = csr_array((data, sp_mat.indices.copy(), sp_mat.indptr.copy()), shape=sp_mat.shape)
        else:
            sp = sp_mat
        return sparse_dense_matmul(sp, orth_mat, device=device)

    if norm == "log":
        data = (
            np.ones_like(sp_mat.data, dtype=work_dtype)
            if bin_mode
            else np.log10(sp_mat.data).astype(work_dtype)
        )
        sp = csr_array((data, sp_mat.indices.copy(), sp_mat.indptr.copy()), shape=sp_mat.shape)
        return sparse_dense_matmul(sp, orth_mat, device=device)

    # norm in ("Neg", "logNeg")
    # Exact algebraic identity: X_tilde @ B == X_hat_sparse @ B - 1_N @ (1_L^T @ B)
    if bin_mode:
        hat_data = np.full_like(sp_mat.data, 2.0, dtype=work_dtype)
    elif norm == "Neg":
        hat_data = (sp_mat.data + 1.0).astype(work_dtype)
    else:  # norm == "logNeg"
        hat_data = (np.log10(sp_mat.data) + 1.0).astype(work_dtype)

    hat_x = csr_array((hat_data, sp_mat.indices.copy(), sp_mat.indptr.copy()), shape=sp_mat.shape)
    sparse_term = sparse_dense_matmul(hat_x, orth_mat, device=device)
    col_sum_b = np.sum(orth_mat, axis=0, keepdims=True, dtype=work_dtype)
    return sparse_term - col_sum_b


def sweep(
    input: str
    | Path
    | dict[str, str]
    | Sequence[str]
    | NDArray[np.floating[Any]]
    | csr_array
    | Any,
    orthbase: OrthBaseResult | NDArray[np.floating[Any]] | None = None,
    bin_mode: bool = False,
    norm: NormMode = "none",
    ncores: int | None = None,
    extension: str | list[str] = "",
    mask: Sequence[int] | NDArray[np.int32] | None = None,
    seqtype: SeqType = "AA",
    low_ram_mode: bool = False,
    concatenate: bool = False,
    transpose: bool = False,
    rna_seq_data: bool = False,
    device: DeviceType = "auto",
    verbose: bool = True,
    dtype: DTypeLike = np.float32,
) -> SweepResult:
    """Spaced Words Projection (SWeeP) dimensionality reduction.

    Projects biological sequences or numerical matrices into compact vectors
    with exact numerical and mathematical parity to rSWeeP.
    """
    start_time = time.perf_counter()

    sw_checks("norm", norm)
    sw_checks("bin", bin_mode)
    sw_checks("transpose", transpose)
    sw_checks("RNAseqdata", rna_seq_data)
    sw_checks("concatenate", concatenate)
    sw_checks("lowRAMmode", low_ram_mode)
    sw_checks("ncores", ncores)
    ncores = n_cores_def(ncores)
    target_device = get_available_device(device)
    target_dtype = np.dtype(dtype)

    headers: list[str] | None = None
    bmask: NDArray[np.int32] | None = None
    features: list[str] | None = None
    samples: list[str] | None = None

    # Check if input is a numerical matrix
    if isinstance(input, (np.ndarray, sparray, spmatrix)):
        mat = input.T if (transpose or rna_seq_data) else input
        n_rows, n_cols = mat.shape
        samples = [f"sample_{i + 1}" for i in range(n_rows)]
        features = [f"feature_{j + 1}" for j in range(n_cols)]

        if orthbase is None:
            orth_res = orth_base(lin=n_cols, col=600, dtype=target_dtype)
            b_mat = orth_res.mat
        elif isinstance(orthbase, OrthBaseResult):
            b_mat = orthbase.mat
        else:
            b_mat = np.asarray(orthbase, dtype=target_dtype)

        sw_check_orthbase(b_mat, mat)

        if isinstance(mat, (sparray, spmatrix)):
            sp_mat = mat if isinstance(mat, csr_array) else csr_array(mat)
            proj = _project_sparse_matrix(
                sp_mat, b_mat, bin_mode=bin_mode, norm=norm, device=target_device
            )
        else:
            mat_dense = np.asarray(mat, dtype=b_mat.dtype)
            mat_transformed = make_log(mat_dense, bin_mode=bin_mode, norm=norm)
            proj = dense_dense_matmul(mat_transformed, b_mat, device=target_device)

        proj_size = b_mat.shape[1]
        is_matrix = True
    else:
        is_matrix = False
        # Biological sequences
        bmask = def_mask(mask, seqtype)
        kk = 20 if seqtype == "AA" else 4
        lenmax = int(kk ** np.sum(bmask))

        if orthbase is None:
            orth_res = orth_base(
                lin=lenmax, col=600, seqtype=seqtype, mask=bmask, dtype=target_dtype
            )
            b_mat = orth_res.mat
        elif isinstance(orthbase, OrthBaseResult):
            b_mat = orthbase.mat
        else:
            b_mat = np.asarray(orthbase, dtype=target_dtype)

        hdv_res = extract_hdv(
            input,
            mask=bmask,
            seqtype=seqtype,
            bin_mode=bin_mode,
            concatenate=concatenate,
            extension=extension,
            verbose=verbose,
        )

        headers = hdv_res.info.headers
        sw_check_orthbase(b_mat, hdv_res.hdv)

        proj = _project_sparse_matrix(
            hdv_res.hdv, b_mat, bin_mode=bin_mode, norm=norm, device=target_device
        )
        proj_size = b_mat.shape[1]

    elapsed = time.perf_counter() - start_time
    info = SweepInfo(
        projection_size=proj_size,
        bin="binary (TRUE)" if bin_mode else "counting (FALSE)",
        norm=norm,
        version=DEFAULT_VERSION,
        time_elapsed=elapsed,
        headers=headers,
        mask=bmask,
        sequence_type=seqtype if not is_matrix else None,
        extension=extension if not is_matrix else None,
        concatenate=concatenate if not is_matrix else None,
        transpose=transpose if is_matrix else None,
        rna_seq_data=rna_seq_data if is_matrix else None,
        features=features,
        samples=samples,
    )

    return SweepResult(proj=proj, info=info)


def sweep_lite(
    input: str
    | Path
    | dict[str, str]
    | Sequence[str]
    | NDArray[np.floating[Any]]
    | csr_array
    | Any,
    psz: int = 1369,
    bin_mode: bool = False,
    norm: NormMode = "none",
    ncores: int | None = None,
    extension: str | list[str] = "",
    mask: Sequence[int] | NDArray[np.int32] | None = None,
    seqtype: SeqType = "AA",
    low_ram_mode: bool = True,
    concatenate: bool = False,
    transpose: bool = False,
    rna_seq_data: bool = False,
    nk: int = 15000,
    device: DeviceType = "auto",
    verbose: bool = True,
    dtype: DTypeLike = np.float32,
) -> SweepResult:
    """Spaced Words Projection version lite (SWeePlite).

    Optimized for large biological sequence sets and generic matrices with low RAM footprint.
    """
    sw_checks("psz", psz)
    sw_checks("bin", bin_mode)
    sw_checks("norm", norm)

    if isinstance(input, (np.ndarray, sparray, spmatrix)):
        mat = input.T if (transpose or rna_seq_data) else input
        orth = orth_base(lin=mat.shape[1], col=psz, dtype=dtype)
        return sweep(
            input,
            orthbase=orth,
            bin_mode=bin_mode,
            norm=norm,
            ncores=ncores,
            transpose=transpose,
            rna_seq_data=rna_seq_data,
            device=device,
            verbose=verbose,
            dtype=dtype,
        )

    bmask = def_mask(mask, seqtype)
    kk = 20 if seqtype == "AA" else 4
    lenmax = int(kk ** np.sum(bmask))

    orth = orth_base(lin=lenmax, col=psz, seqtype=seqtype, mask=bmask, dtype=dtype)
    return sweep(
        input,
        orthbase=orth,
        bin_mode=bin_mode,
        norm=norm,
        ncores=ncores,
        extension=extension,
        mask=bmask,
        seqtype=seqtype,
        low_ram_mode=low_ram_mode,
        concatenate=concatenate,
        device=device,
        verbose=verbose,
        dtype=dtype,
    )
