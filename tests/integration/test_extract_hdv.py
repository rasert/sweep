from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_array

from sweep.hdv import extract_hdv
from sweep.models import HDVResult
from tests.helpers.golden import load_golden_csv, load_golden_mtx


def test_extract_hdv_in_memory_sequences() -> None:
    aa_seqs = {
        "seq1": "ACDEFGHIKLMNPQRSTVWY",
        "seq2": "ACDACDACDACD",
    }

    # mask [1, 0, 1] -> 20^2 = 400 positions
    res = extract_hdv(aa_seqs, mask=[1, 0, 1], seqtype="AA", bin_mode=False, verbose=False)

    assert isinstance(res, HDVResult)
    assert isinstance(res.hdv, csr_array)
    assert res.hdv.shape == (2, 400)
    assert res.info.headers == ["seq1", "seq2"]
    np.testing.assert_array_equal(res.info.mask, np.array([1, 0, 1]))
    assert res.info.sequence_type == "AA"
    assert not res.info.concatenate
    assert isinstance(res.info.saturation, np.ndarray)
    assert len(res.info.saturation) == 2
    assert np.all((res.info.saturation > 0.0) & (res.info.saturation <= 1.0))

    # Binary mode
    res_bin = extract_hdv(aa_seqs, mask=[1, 0, 1], seqtype="AA", bin_mode=True, verbose=False)
    assert np.all(np.isin(res_bin.hdv.data, [1.0]))

    # Concatenate mode
    res_cat = extract_hdv(aa_seqs, mask=[1, 0, 1], seqtype="AA", concatenate=True, verbose=False)
    assert res_cat.hdv.shape == (1, 400)
    assert res_cat.info.concatenate


def test_extract_hdv_from_folder_against_golden_files(mito_fasta_dir: Path) -> None:
    # 1. Compact mask [1, 1, 1] -> 400 positions (Golden CSV)
    res_compact = extract_hdv(
        mito_fasta_dir,
        mask=[1, 1, 1],
        seqtype="AA",
        bin_mode=False,
        extension=".faa",
        verbose=False,
    )
    golden_compact = load_golden_csv("hdv_compact_aa_400.csv")
    np.testing.assert_allclose(res_compact.hdv.toarray(), golden_compact, rtol=1e-12)

    # 2. Canonical mask [2, 1, 2] -> 160,000 positions (Golden MTX sparse matrix)
    res_canonical = extract_hdv(
        mito_fasta_dir,
        mask=[2, 1, 2],
        seqtype="AA",
        bin_mode=False,
        extension=".faa",
        verbose=False,
    )
    golden_mtx = load_golden_mtx("hdv_mito_14x160000.mtx")
    assert res_canonical.hdv.shape == (14, 160000)
    np.testing.assert_allclose(res_canonical.hdv.toarray(), golden_mtx.toarray(), rtol=1e-12)
    # Saturation of mitochondrial proteomes is low (~1-5%)
    assert np.all((res_canonical.info.saturation > 0.005) & (res_canonical.info.saturation < 0.10))
