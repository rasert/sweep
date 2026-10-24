from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from sweep.checks import sw_checks
from sweep.indexing import def_mask
from sweep.models import (
    DEFAULT_SEED,
    DEFAULT_VERSION,
    DTypeLike,
    NormMode,
    OrthBaseResult,
    SeqType,
)
from sweep.r_mt19937 import R_MT19937

PRIME_LIST_50: Final[tuple[int, ...]] = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
    229,
)


def n_cores_def(ncores: int | None) -> int:
    """Standardizes parallel core count matching R's NCoresDef."""
    if ncores is None or ncores <= 0:
        return 2
    return ncores


@dataclass(slots=True, frozen=True)
class LiteParam:
    """Precomputed projection parameters for SWeePlite and core_loop."""

    pslist: list[int]
    nps: int
    mproj: NDArray[np.float64]
    xnorm: float


def lite_param(psz: int, lenmax: int, seed: int = DEFAULT_SEED) -> LiteParam:
    """Generates lite projection parameters matching R's liteParam."""
    sw_checks("psz", psz)
    rng = R_MT19937(seed)
    raw = rng.runif(len(PRIME_LIST_50) * psz)
    mproj = raw.reshape((len(PRIME_LIST_50), psz), order="F")
    xnorm = float(np.sqrt(lenmax / 3.0))
    return LiteParam(
        pslist=list(PRIME_LIST_50),
        nps=len(PRIME_LIST_50),
        mproj=mproj,
        xnorm=xnorm,
    )


def orth_base(
    lin: int | None = None,
    col: int | None = None,
    seqtype: SeqType = "AA",
    mask: Sequence[int] | NDArray[np.int32] | None = None,
    seed: int = DEFAULT_SEED,
    dtype: DTypeLike = np.float32,
    chunk_size: int = 8192,
) -> OrthBaseResult:
    """Generates an orthonormal basis matrix for SWeeP projection.

    Uses cache-friendly chunked float64 internal arithmetic to guarantee exact
    numerical phase stability, writing into the requested target dtype (default float32).
    """
    if col is None:
        msg = "'col' (projection size) must be specified."
        raise ValueError(msg)

    sw_checks("psz", col)
    sw_checks("seqtype", seqtype)

    if lin is None:
        bmask = def_mask(mask, seqtype)
        kk = 20 if seqtype == "AA" else 4
        lin = int(kk ** np.sum(bmask))

    target_dtype = np.dtype(dtype)
    if target_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        msg = f"'dtype' must be float32 or float64, got {target_dtype}."
        raise ValueError(msg)

    rng = R_MT19937(seed)
    raw = rng.runif(len(PRIME_LIST_50) * col)
    mproj = raw.reshape((len(PRIME_LIST_50), col), order="F")

    xnorm = float(np.sqrt(lin / 3.0))
    scale = 2.0 / xnorm
    offset = -1.0 / xnorm

    primes = np.array(PRIME_LIST_50, dtype=np.int64)[None, :]
    mat = np.empty((lin, col), dtype=target_dtype)

    step = max(1, min(chunk_size, lin))
    buf_dt = np.empty((step, col), dtype=np.float64)
    buf_floor = np.empty((step, col), dtype=np.float64)
    buf_mret = np.empty((step, len(PRIME_LIST_50)), dtype=np.float64)

    for start in range(0, lin, step):
        end = min(start + step, lin)
        n_rows = end - start
        sub_idx = np.arange(start + 1, end + 1, dtype=np.int64)[:, None]
        mret_view = buf_mret[:n_rows]
        np.mod(sub_idx, primes, out=mret_view, casting="unsafe")
        mret_view += 1.0

        dt_view = buf_dt[:n_rows]
        floor_view = buf_floor[:n_rows]

        np.matmul(mret_view, mproj, out=dt_view)
        np.floor(dt_view, out=floor_view)
        dt_view -= floor_view
        dt_view *= scale
        dt_view += offset

        mat[start:end] = dt_view

    return OrthBaseResult(mat=mat, seed=seed, version=DEFAULT_VERSION)


def core_loop(
    xnorm: float,
    psz: int,
    Mproj: NDArray[np.float64],
    pslist: Sequence[int] | NDArray[np.int64],
    nps: int,
    hdv_idx: NDArray[np.int64],
    hdv_count: NDArray[np.float64] | None,
    nk: int = 1000,
    bin_mode: bool = False,
    norm: NormMode = "none",
    dtype: DTypeLike = np.float32,
) -> NDArray[np.floating[Any]]:
    """Projects sparse HDV sequence directly without materializing full basis matrix."""
    target_dtype = np.dtype(dtype)
    if len(hdv_idx) == 0:
        return np.zeros(psz, dtype=target_dtype)

    tot = len(hdv_idx)
    primes = np.array(pslist, dtype=np.int64)[None, :]
    out = np.zeros(psz, dtype=np.float64)

    for start in range(0, tot, nk):
        end = min(start + nk, tot)
        sub_idx = hdv_idx[start:end][:, None]
        mret = sub_idx % primes
        dt = (1.0 + mret) @ Mproj
        bs = 2.0 * (dt - np.floor(dt)) - 1.0

        if (not bin_mode or norm in ("Neg", "logNeg")) and hdv_count is not None:
            weights = hdv_count[start:end][:, None]
            bs = weights * bs

        out += np.sum(bs, axis=0) / xnorm

    return out.astype(target_dtype)
