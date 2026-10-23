from __future__ import annotations

import numpy as np
import pytest

from sweep.checks import sw_check_orthbase, sw_checks


def test_sw_checks_validates_seqtype() -> None:
    sw_checks("seqtype", "AA")
    sw_checks("seqtype", "NT")

    with pytest.raises(ValueError, match="'seqtype' should be any one of the options"):
        sw_checks("seqtype", "DNA")

    with pytest.raises(ValueError, match="'seqtype' should be any one of the options"):
        sw_checks("seqtype", ["AA", "NT"])

    with pytest.raises(ValueError, match="'seqtype' should be any one of the options"):
        sw_checks("seqtype", 123)


def test_sw_checks_validates_extension() -> None:
    sw_checks("extension", ".faa")
    sw_checks("extension", [".faa", ".fasta"])

    with pytest.raises(ValueError, match="'extension' should be a character"):
        sw_checks("extension", 123)


def test_sw_checks_validates_psz() -> None:
    sw_checks("psz", 600)
    sw_checks("psz", 1369)

    with pytest.raises(ValueError, match="'psz' should be a integer numeric value"):
        sw_checks("psz", 600.5)

    with pytest.raises(ValueError, match="'psz' should be a integer numeric value"):
        sw_checks("psz", "600")

    with pytest.raises(ValueError, match="'psz' should be a integer numeric value"):
        sw_checks("psz", True)


def test_sw_checks_validates_mask() -> None:
    sw_checks("mask", [2, 1, 2])
    sw_checks("mask", [5, 5, 5])
    sw_checks("mask", np.array([2, 1, 2]))

    with pytest.raises(ValueError, match="'mask' should be a vector"):
        sw_checks("mask", "11011")

    with pytest.raises(ValueError, match="'mask' should be a vector"):
        sw_checks("mask", [])


def test_sw_checks_validates_boolean_flags() -> None:
    for flag_name in (
        "bin",
        "readEachFASTA",
        "lowRAMmode",
        "RNAseqdata",
        "transpose",
        "concatenate",
    ):
        sw_checks(flag_name, True)
        sw_checks(flag_name, False)

        with pytest.raises(ValueError, match="should be logical"):
            sw_checks(flag_name, "TRUE")

        with pytest.raises(ValueError, match="should be logical"):
            sw_checks(flag_name, 1)


def test_sw_checks_validates_ncores_and_fastalist() -> None:
    sw_checks("ncores", None)
    sw_checks("ncores", 2)

    with pytest.raises(ValueError, match="'ncores' should be a positive integer"):
        sw_checks("ncores", 0)

    with pytest.raises(ValueError, match="'ncores' should be a positive integer"):
        sw_checks("ncores", -1)

    with pytest.raises(ValueError, match="'ncores' should be a positive integer"):
        sw_checks("ncores", True)

    sw_checks("fastalist", ["a.faa", "b.faa"])
    with pytest.raises(ValueError, match="The folder provided does not seem to contain files"):
        sw_checks("fastalist", [])


def test_sw_checks_validates_norm_modes() -> None:
    for m in ("none", "log", "Neg", "logNeg"):
        sw_checks("norm", m)

    with pytest.raises(ValueError, match="'norm' must be one of the following options"):
        sw_checks("norm", "invalid")


def test_sw_check_orthbase_validates_dimensions() -> None:
    orth = np.arange(20, dtype=np.float64).reshape(5, 4)
    input_mat = np.arange(15, dtype=np.float64).reshape(3, 5)

    # Valid: orth.shape[0] == input_mat.shape[1] (5 == 5)
    sw_check_orthbase(orth, input_mat)

    # Incompatible dimensions: orth.shape[0] (5) != input_err.shape[1] (4)
    input_err = np.arange(12, dtype=np.float64).reshape(3, 4)
    with pytest.raises(ValueError, match="The number of rows of the orthbase matrix must be equal"):
        sw_check_orthbase(orth, input_err)

    # Non-2D array
    with pytest.raises(ValueError, match="Orthbase must be an array"):
        sw_check_orthbase([1, 2, 3], input_mat)
