from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from sweep.checks import sw_checks
from sweep.indexing import concatena_each, def_mask, return_pot, seq2num
from sweep.models import DEFAULT_VERSION, HDVInfo, HDVResult, NormMode, SeqType


def make_log(
    input: NDArray[np.float64] | csr_array,
    bin_mode: bool = False,
    norm: NormMode = "none",
) -> NDArray[np.float64] | csr_array:
    """Transforms dense or sparse matrix according to binary and normalization rules."""
    sw_checks("bin", bin_mode)
    sw_checks("norm", norm)

    if isinstance(input, csr_array):
        if norm in ("Neg", "logNeg"):
            # Construct dense array without calling .toarray() or .todense()
            dense_res = np.full(input.shape, -1.0, dtype=np.float64)
            for r in range(input.shape[0]):
                start = int(input.indptr[r])
                end = int(input.indptr[r + 1])
                cols = input.indices[start:end]
                vals = input.data[start:end]
                if bin_mode:
                    dense_res[r, cols] = 1.0
                elif norm == "logNeg":
                    dense_res[r, cols] = np.log10(vals)
                else:
                    dense_res[r, cols] = vals
            return dense_res

        if norm == "none":
            if bin_mode:
                data = np.ones_like(input.data, dtype=np.float64)
                return csr_array(
                    (data, input.indices.copy(), input.indptr.copy()),
                    shape=input.shape,
                )
            return input.copy()

        # norm == "log"
        data = np.ones_like(input.data, dtype=np.float64) if bin_mode else np.log10(input.data)
        return csr_array(
            (data, input.indices.copy(), input.indptr.copy()),
            shape=input.shape,
        )

    res = np.array(input, dtype=np.float64, copy=True)
    if bin_mode:
        res[res != 0] = 1.0

    if norm == "log" and not bin_mode:
        nonzero = res != 0
        res[nonzero] = np.log10(res[nonzero])
    elif norm == "logNeg":
        if bin_mode:
            res[res == 0] = -1.0
        else:
            nonzero = res != 0
            res[nonzero] = np.log10(res[nonzero])
            res[~nonzero] = -1.0
    elif norm == "Neg":
        res[res == 0] = -1.0

    return res


def read_mask_vec(
    seq: NDArray[np.int32],
    mask: Sequence[int] | NDArray[np.int32],
    seqtype: SeqType,
    bin_mode: bool = False,
    norm: NormMode = "none",
    lenmax: int | None = None,
) -> NDArray[np.float64]:
    """Calculates dense HDV count or normalized vector for a single sequence."""
    sw_checks("seqtype", seqtype)
    sw_checks("bin", bin_mode)
    sw_checks("norm", norm)
    bmask = def_mask(mask, seqtype)

    if lenmax is None:
        kk = 20 if seqtype == "AA" else 4
        lenmax = int(kk ** np.sum(bmask))

    pot = return_pot(seq, mask, seqtype)
    outseq = np.zeros(lenmax, dtype=np.float64)
    if len(pot) > 0:
        counts = np.bincount(pot, minlength=lenmax + 1)[1 : lenmax + 1].astype(np.float64)
        outseq = counts

    if bin_mode:
        nonzero = outseq != 0
        outseq[nonzero] = 1.0
        if norm in ("Neg", "logNeg"):
            outseq[~nonzero] = -1.0
    else:
        if norm == "log":
            nonzero = outseq != 0
            outseq[nonzero] = np.log10(outseq[nonzero])
        elif norm == "Neg":
            outseq[outseq == 0] = -1.0
        elif norm == "logNeg":
            nonzero = outseq != 0
            outseq[nonzero] = np.log10(outseq[nonzero])
            outseq[~nonzero] = -1.0

    return outseq


def read_mask(
    seq: NDArray[np.int32],
    mask: Sequence[int] | NDArray[np.int32],
    seqtype: SeqType,
    bin_mode: bool = False,
    norm: NormMode = "none",
    lenmax: int | None = None,
) -> dict[str, Any]:
    """Calculates non-zero indices and values for a single sequence matching R's readmask."""
    sw_checks("seqtype", seqtype)
    sw_checks("bin", bin_mode)
    sw_checks("norm", norm)
    bmask = def_mask(mask, seqtype)

    if lenmax is None:
        kk = 20 if seqtype == "AA" else 4
        lenmax = int(kk ** np.sum(bmask))

    pot = return_pot(seq, mask, seqtype)
    counts = np.zeros(lenmax, dtype=np.float64)
    if len(pot) > 0:
        counts = np.bincount(pot, minlength=lenmax + 1)[1 : lenmax + 1].astype(np.float64)

    nz_idx = np.where(counts != 0)[0].astype(np.int64)

    out: dict[str, Any] = {"idx": nz_idx, "count": None}

    if bin_mode:
        if norm in ("none", "log"):
            out["count"] = None
        elif norm in ("Neg", "logNeg"):
            count_arr = np.full(lenmax, -1.0, dtype=np.float64)
            count_arr[nz_idx] = 1.0
            out["count"] = count_arr
            out["idx"] = np.arange(lenmax, dtype=np.int64)
    else:
        if norm == "none":
            out["count"] = counts[nz_idx]
        elif norm == "log":
            out["count"] = np.log10(counts[nz_idx])
        elif norm == "Neg":
            count_arr = np.full(lenmax, -1.0, dtype=np.float64)
            count_arr[nz_idx] = counts[nz_idx]
            out["count"] = count_arr
            out["idx"] = np.arange(lenmax, dtype=np.int64)
        elif norm == "logNeg":
            count_arr = np.full(lenmax, -1.0, dtype=np.float64)
            count_arr[nz_idx] = np.log10(counts[nz_idx])
            out["count"] = count_arr
            out["idx"] = np.arange(lenmax, dtype=np.int64)

    return out


def extract_hdv(
    input: str | Path | dict[str, str] | Sequence[str],
    mask: Sequence[int] | NDArray[np.int32] | None = None,
    seqtype: SeqType = "AA",
    bin_mode: bool = False,
    concatenate: bool = False,
    extension: str | list[str] = "",
    verbose: bool = True,
) -> HDVResult:
    """Extracts High Dimensional Vectors into a scipy CSR sparse array."""
    start_time = time.perf_counter()

    sw_checks("seqtype", seqtype)
    sw_checks("bin", bin_mode)
    sw_checks("concatenate", concatenate)
    bmask = def_mask(mask, seqtype)

    kk = 20 if seqtype == "AA" else 4
    lenmax = int(kk ** np.sum(bmask))

    headers: list[str] = []
    seq_strings: list[str] = []

    if isinstance(input, (str, Path)):
        in_path = Path(input)
        if in_path.is_dir():
            if isinstance(extension, (str, list)):
                sw_checks("extension", extension)
            ext_list: list[str] = (
                [extension]
                if isinstance(extension, str) and extension
                else (extension if isinstance(extension, list) else [])
            )
            matched_files: list[Path] = [
                f
                for f in in_path.rglob("*")
                if f.is_file() and (not ext_list or any(f.name.endswith(ext) for ext in ext_list))
            ]
            matched_files.sort(key=lambda p: str(p))
            sw_checks("fastalist", [str(f) for f in matched_files])

            headers = [f.stem for f in matched_files]
            spacer = "*" * len(bmask)
            seq_strings = [concatena_each(f, spacer) for f in matched_files]
        else:
            msg = f"Path {in_path} is not a directory."
            raise ValueError(msg)
    elif isinstance(input, dict):
        headers = list(input.keys())
        seq_strings = list(input.values())
        if concatenate:
            spacer = "*" * len(bmask)
            seq_strings = [spacer.join(seq_strings)]
            headers = headers[:1]
    elif isinstance(input, Sequence) and not isinstance(input, str):
        headers = [f"seq_{i + 1}" for i in range(len(input))]
        seq_strings = list(input)
        if concatenate:
            spacer = "*" * len(bmask)
            seq_strings = [spacer.join(seq_strings)]
            headers = headers[:1]
    else:
        msg = f"Unsupported input type: {type(input)}"
        raise TypeError(msg)

    n_seqs = len(seq_strings)
    all_indices: list[NDArray[np.int32]] = []
    all_data: list[NDArray[np.float64]] = []
    indptr: list[int] = [0]

    for k in range(n_seqs):
        if verbose:
            print(f"starting sequence {k + 1} of {n_seqs}", end="", flush=True)

        seq_num = seq2num(seq_strings[k], seqtype)
        pot = return_pot(seq_num, bmask, seqtype)

        if len(pot) > 0:
            uniq_vals, counts = np.unique(pot, return_counts=True)
            cols = (uniq_vals - 1).astype(np.int32)
            row_data = (
                np.ones(len(cols), dtype=np.float64) if bin_mode else counts.astype(np.float64)
            )
        else:
            cols = np.array([], dtype=np.int32)
            row_data = np.array([], dtype=np.float64)

        all_indices.append(cols)
        all_data.append(row_data)
        indptr.append(indptr[-1] + len(cols))

        if verbose:
            print(" - complete")

    final_indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)
    final_data = np.concatenate(all_data) if all_data else np.array([], dtype=np.float64)
    final_indptr = np.array(indptr, dtype=np.int32)

    hdv = csr_array((final_data, final_indices, final_indptr), shape=(n_seqs, lenmax))

    nnz_per_row = np.diff(hdv.indptr)
    if n_seqs > 1:
        saturation: NDArray[np.float64] | float = (nnz_per_row / float(lenmax)).astype(np.float64)
    else:
        saturation = float(nnz_per_row[0] / float(lenmax))

    elapsed = time.perf_counter() - start_time
    info = HDVInfo(
        headers=headers,
        mask=bmask,
        sequence_type=seqtype,
        extension=extension,
        concatenate=concatenate,
        bin="binary (TRUE)" if bin_mode else "counting (FALSE)",
        version=DEFAULT_VERSION,
        saturation=saturation,
        time_elapsed=elapsed,
    )

    return HDVResult(hdv=hdv, info=info)
