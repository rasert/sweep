from __future__ import annotations

import numpy as np

from sweep.models import DEFAULT_SEED
from sweep.r_mt19937 import R_MT19937
from tests.helpers.golden import load_golden_csv


def test_r_mt19937_first_numbers_match_r_runif() -> None:
    rng = R_MT19937(DEFAULT_SEED)
    vals = rng.runif(3)
    expected = np.array([0.34381698770448565, 0.2275575001258403, 0.909247390460223])
    np.testing.assert_allclose(vals, expected, rtol=1e-14, atol=1e-14)


def test_r_mt19937_mproj_50x10_matches_golden() -> None:
    rng = R_MT19937(DEFAULT_SEED)
    raw_vals = rng.runif(50 * 10)
    mproj = raw_vals.reshape((50, 10), order="F")
    golden = load_golden_csv("mproj_50x10_seed647474747.csv")
    np.testing.assert_allclose(mproj, golden, rtol=1e-12, atol=1e-12)


def test_r_mt19937_mproj_50x600_matches_golden() -> None:
    rng = R_MT19937(DEFAULT_SEED)
    raw_vals = rng.runif(50 * 600)
    mproj = raw_vals.reshape((50, 600), order="F")
    golden = load_golden_csv("mproj_50x600_seed647474747.csv")
    np.testing.assert_allclose(mproj, golden, rtol=1e-12, atol=1e-12)
