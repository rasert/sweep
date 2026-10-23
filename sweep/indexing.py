from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
from Bio import SeqIO
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from sweep.checks import sw_checks
from sweep.models import SeqType

AA_MAP: Final[dict[str, int]] = {
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "B": 21,
    "Z": 22,
    "X": 23,
    "*": 24,
    "-": 25,
    "U": 26,
    "O": 27,
    "J": 28,
    "?": 0,
}

NT_MAP: Final[dict[str, int]] = {
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "U": 4,
    "R": 5,
    "Y": 6,
    "K": 7,
    "M": 8,
    "S": 9,
    "W": 10,
    "B": 11,
    "D": 12,
    "H": 13,
    "V": 14,
    "N": 15,
    "-": 16,
    "*": 0,
}


def convert_mask(x: Sequence[int]) -> NDArray[np.int32]:
    """Converts run-length encoded mask (e.g. [2, 1, 2]) to binary form (e.g. [1, 1, 0, 1, 1])."""
    n = len(x)
    y = [(i % 2) for i in range(1, n + 1)]
    expanded: list[int] = []
    for count, val in zip(x, y, strict=True):
        expanded.extend([val] * int(count))
    return np.array(expanded, dtype=np.int32)


def def_mask(mask: Sequence[int] | NDArray[np.int32] | None, seqtype: SeqType) -> NDArray[np.int32]:
    """Resolves default or user-provided mask into binary form."""
    sw_checks("seqtype", seqtype)
    if mask is None:
        if seqtype == "AA":
            return np.array([1, 1, 0, 1, 1], dtype=np.int32)
        return np.array([1] * 5 + [0] * 5 + [1] * 5, dtype=np.int32)

    sw_checks("mask", mask)
    mask_arr = np.array(mask, dtype=np.int32)
    if np.max(mask_arr) > 1:
        return convert_mask(mask_arr.tolist())
    return mask_arr


def seq2num(seq: str, seqtype: SeqType) -> NDArray[np.int32]:
    """Maps sequence string to canonical integer identifiers."""
    sw_checks("seqtype", seqtype)
    mapping = AA_MAP if seqtype == "AA" else NT_MAP
    upper_seq = seq.upper()
    return np.array([mapping.get(ch, 0) for ch in upper_seq], dtype=np.int32)


def return_pot(
    seq_num: NDArray[np.int32],
    mask: Sequence[int] | NDArray[np.int32],
    seqtype: SeqType,
) -> NDArray[np.int64]:
    """Calculates 1-based vigesimal or quaternary indices f(S_k) for spaced-word windows."""
    bmask = def_mask(mask, seqtype)
    lmer = len(bmask)
    m = int(np.sum(bmask))
    kk = 20 if seqtype == "AA" else 4

    nwindows = len(seq_num) - lmer + 1
    if nwindows <= 0:
        return np.array([], dtype=np.int64)

    # O(1) memory view of all overlapping windows
    windows = sliding_window_view(seq_num, window_shape=lmer)
    active_cols = np.where(bmask != 0)[0]
    maskwin = windows[:, active_cols]

    # Strict invalidation: all active positions must be in canonical range [1, kk]
    valid_mask = np.all((maskwin >= 1) & (maskwin <= kk), axis=1)
    if not np.any(valid_mask):
        return np.array([], dtype=np.int64)

    vals = (maskwin[valid_mask] - 1).astype(np.int64)
    powers = kk ** np.arange(m, dtype=np.int64)
    pot = 1 + np.sum(vals * powers, axis=1)
    return pot.astype(np.int64)


def concatena_each(file_path: str | Path, spacer: str) -> str:
    """Reads all sequences from a FASTA file and joins them with the specified spacer."""
    records = list(SeqIO.parse(str(file_path), "fasta"))
    return spacer.join(str(rec.seq) for rec in records)


def concatena_all(file_list: Sequence[str | Path], spacer: str) -> list[str]:
    """Reads each FASTA file in the list, concatenating internal sequences with spacer."""
    return [concatena_each(f, spacer) for f in file_list]
