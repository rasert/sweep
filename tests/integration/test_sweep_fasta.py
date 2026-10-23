from __future__ import annotations

from pathlib import Path

import numpy as np

from sweep.backend import is_gpu_available
from sweep.core import sweep, sweep_lite
from sweep.models import DEFAULT_SEED, SweepResult
from sweep.projection_core import orth_base
from tests.helpers.golden import load_golden_csv


def test_sweep_fasta_count_none_matches_golden_file(mito_fasta_dir: Path) -> None:
    orth = orth_base(lin=160000, col=600, seed=DEFAULT_SEED)
    res = sweep(
        mito_fasta_dir,
        orthbase=orth,
        bin_mode=False,
        norm="none",
        mask=[2, 1, 2],
        seqtype="AA",
        extension=".faa",
        verbose=False,
    )

    assert isinstance(res, SweepResult)
    assert res.proj.shape == (14, 600)
    assert res.info.projection_size == 600

    golden = load_golden_csv("sweep_mito_aa_14x600_count_none.csv")
    np.testing.assert_allclose(res.proj, golden, rtol=1e-12, atol=1e-12)


def test_sweep_fasta_bin_none_matches_golden_file(mito_fasta_dir: Path) -> None:
    orth = orth_base(lin=160000, col=600, seed=DEFAULT_SEED)
    res = sweep(
        mito_fasta_dir,
        orthbase=orth,
        bin_mode=True,
        norm="none",
        mask=[2, 1, 2],
        seqtype="AA",
        extension=".faa",
        verbose=False,
    )

    golden = load_golden_csv("sweep_mito_aa_14x600_bin_none.csv")
    np.testing.assert_allclose(res.proj, golden, rtol=1e-12, atol=1e-12)


def test_sweep_lite_matches_sweep_on_fasta_folder(mito_fasta_dir: Path) -> None:
    res_lite = sweep_lite(
        mito_fasta_dir,
        psz=600,
        bin_mode=False,
        norm="none",
        mask=[2, 1, 2],
        seqtype="AA",
        extension=".faa",
        verbose=False,
    )

    golden = load_golden_csv("sweep_mito_aa_14x600_count_none.csv")
    np.testing.assert_allclose(res_lite.proj, golden, rtol=1e-12, atol=1e-12)


def test_sweep_gpu_cpu_equivalence_on_fasta(mito_fasta_dir: Path) -> None:
    if not is_gpu_available():
        return

    orth = orth_base(lin=160000, col=600, seed=DEFAULT_SEED)
    res_cpu = sweep(
        mito_fasta_dir,
        orthbase=orth,
        bin_mode=False,
        norm="none",
        mask=[2, 1, 2],
        seqtype="AA",
        extension=".faa",
        device="cpu",
        verbose=False,
    )
    res_gpu = sweep(
        mito_fasta_dir,
        orthbase=orth,
        bin_mode=False,
        norm="none",
        mask=[2, 1, 2],
        seqtype="AA",
        extension=".faa",
        device="cuda",
        verbose=False,
    )

    np.testing.assert_allclose(res_gpu.proj, res_cpu.proj, rtol=1e-12, atol=1e-12)
