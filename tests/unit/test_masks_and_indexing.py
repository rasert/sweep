from __future__ import annotations

import numpy as np

from sweep.indexing import convert_mask, def_mask, return_pot, seq2num


def test_convert_mask_and_def_mask_binary_expansion() -> None:
    # Canonical mask for proteins: [2, 1, 2] -> [1, 1, 0, 1, 1]
    expected_aa = np.array([1, 1, 0, 1, 1], dtype=np.int32)
    np.testing.assert_array_equal(convert_mask([2, 1, 2]), expected_aa)

    # Mask for nucleotides: [5, 5, 5] -> 5 ones, 5 zeros, 5 ones
    expected_nt = np.array([1] * 5 + [0] * 5 + [1] * 5, dtype=np.int32)
    np.testing.assert_array_equal(convert_mask([5, 5, 5]), expected_nt)

    # def_mask with None defaults
    np.testing.assert_array_equal(def_mask(None, "AA"), np.array([1, 1, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(def_mask(None, "NT"), expected_nt)

    # def_mask with binary mask already
    np.testing.assert_array_equal(def_mask([1, 0, 1], "AA"), np.array([1, 0, 1], dtype=np.int32))

    # def_mask keeps max(mask) <= 1 unchanged
    np.testing.assert_array_equal(def_mask([1, 1, 1], "AA"), np.array([1, 1, 1], dtype=np.int32))


def test_seq2num_maps_alphabets_and_special_characters() -> None:
    # Canonical 20 amino acids -> 1..20
    aa_canon = "ARNDCQEGHILKMFPSTWYV"
    np.testing.assert_array_equal(seq2num(aa_canon, "AA"), np.arange(1, 21, dtype=np.int32))

    # Special / ambiguous characters in AA:
    # B=21, Z=22, X=23, *=24, -=25, U=26, O=27, J=28, ?=0
    aa_special = "BZX*-UOJ?"
    expected_special = np.array([21, 22, 23, 24, 25, 26, 27, 28, 0], dtype=np.int32)
    np.testing.assert_array_equal(seq2num(aa_special, "AA"), expected_special)

    # Case insensitivity
    np.testing.assert_array_equal(seq2num("arnd", "AA"), np.array([1, 2, 3, 4], dtype=np.int32))

    # Nucleotides: A=1, C=2, G=3, T=4, U=4
    nt_test = "ACGTU"
    np.testing.assert_array_equal(seq2num(nt_test, "NT"), np.array([1, 2, 3, 4, 4], dtype=np.int32))
    np.testing.assert_array_equal(seq2num("N*", "NT"), np.array([15, 0], dtype=np.int32))


def test_return_pot_calculates_vigesimal_and_quaternary_indices() -> None:
    # AA with mask [1, 0, 1] on "ACD": k-mers "A.D"
    # A=1 (val-1=0), C=2, D=4 (val-1=3) -> pot = 1 + 0*1 + 3*20 = 61
    seq_num = seq2num("ACD", "AA")
    pots = return_pot(seq_num, [1, 0, 1], "AA")
    np.testing.assert_array_equal(pots, np.array([61], dtype=np.int64))

    # Homogeneous "AAAA" with mask [1, 1] (k=2) -> 1 + 0 + 0 = 1 (3 windows)
    seq_aaaa = seq2num("AAAA", "AA")
    expected_aaaa = np.array([1, 1, 1], dtype=np.int64)
    np.testing.assert_array_equal(return_pot(seq_aaaa, [1, 1], "AA"), expected_aaaa)

    # Max residue "VV" with mask [1, 1] (k=2, V=20, val-1=19) -> 1 + 19 + 19*20 = 400
    seq_vv = seq2num("VV", "AA")
    np.testing.assert_array_equal(return_pot(seq_vv, [1, 1], "AA"), np.array([400], dtype=np.int64))

    # Spacer '*' (val=24) and '?' (val=0) in active positions are filtered out
    # On "AC*DE" with mask [1, 0, 1], "AC*" and "*DE" are invalid; only "C*D" survives
    seq_with_spacer = seq2num("AC*DE", "AA")
    pots_spacer = return_pot(seq_with_spacer, [1, 0, 1], "AA")
    np.testing.assert_array_equal(pots_spacer, np.array([65], dtype=np.int64))
