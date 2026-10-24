from __future__ import annotations

from pathlib import Path

import numpy as np

from sweep.backend import is_gpu_available
from sweep.core import sweep, sweep_lite
from sweep.models import DEFAULT_SEED, SweepResult
from sweep.projection_core import orth_base
from tests.helpers.golden import load_golden_csv


def test_sweep_fasta_count_none_matches_golden_file(mito_fasta_dir: Path) -> None:
    # 1. Default float32 mode
    orth32 = orth_base(lin=160000, col=600, seed=DEFAULT_SEED)
    assert orth32.mat.dtype == np.float32

    res32 = sweep(
        mito_fasta_dir,
        orthbase=orth32,
        bin_mode=False,
        norm="none",
        mask=[2, 1, 2],
        seqtype="AA",
        extension=".faa",
        verbose=False,
    )

    assert isinstance(res32, SweepResult)
    assert res32.proj.shape == (14, 600)
    assert res32.proj.dtype == np.float32
    assert res32.info.projection_size == 600

    golden = load_golden_csv("sweep_mito_aa_14x600_count_none.csv")
    np.testing.assert_allclose(res32.proj, golden, rtol=1e-5, atol=1e-5)

    # 2. Strict float64 mode
    orth64 = orth_base(lin=160000, col=600, seed=DEFAULT_SEED, dtype=np.float64)
    assert orth64.mat.dtype == np.float64

    res64 = sweep(
        mito_fasta_dir,
        orthbase=orth64,
        bin_mode=False,
        norm="none",
        mask=[2, 1, 2],
        seqtype="AA",
        extension=".faa",
        verbose=False,
        dtype=np.float64,
    )
    assert res64.proj.dtype == np.float64
    np.testing.assert_allclose(res64.proj, golden, rtol=1e-12, atol=1e-12)


def test_sweep_fasta_bin_none_matches_golden_file(mito_fasta_dir: Path) -> None:
    orth = orth_base(lin=160000, col=600, seed=DEFAULT_SEED, dtype=np.float64)
    res = sweep(
        mito_fasta_dir,
        orthbase=orth,
        bin_mode=True,
        norm="none",
        mask=[2, 1, 2],
        seqtype="AA",
        extension=".faa",
        verbose=False,
        dtype=np.float64,
    )

    golden = load_golden_csv("sweep_mito_aa_14x600_bin_none.csv")
    np.testing.assert_allclose(res.proj, golden, rtol=1e-12, atol=1e-12)


def test_sweep_lite_matches_sweep_on_fasta_folder(mito_fasta_dir: Path) -> None:
    # 1. Default float32 mode
    res_lite_32 = sweep_lite(
        mito_fasta_dir,
        psz=600,
        bin_mode=False,
        norm="none",
        mask=[2, 1, 2],
        seqtype="AA",
        extension=".faa",
        verbose=False,
    )
    assert res_lite_32.proj.dtype == np.float32

    golden = load_golden_csv("sweep_mito_aa_14x600_count_none.csv")
    np.testing.assert_allclose(res_lite_32.proj, golden, rtol=1e-5, atol=1e-5)

    # 2. Strict float64 mode
    res_lite_64 = sweep_lite(
        mito_fasta_dir,
        psz=600,
        bin_mode=False,
        norm="none",
        mask=[2, 1, 2],
        seqtype="AA",
        extension=".faa",
        verbose=False,
        dtype=np.float64,
    )
    assert res_lite_64.proj.dtype == np.float64
    np.testing.assert_allclose(res_lite_64.proj, golden, rtol=1e-12, atol=1e-12)


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

    np.testing.assert_allclose(res_gpu.proj, res_cpu.proj, rtol=1e-5, atol=1e-5)
