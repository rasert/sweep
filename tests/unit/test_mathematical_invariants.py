from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.sparse import csr_array

from sweep.core import sweep
from sweep.models import DEFAULT_SEED
from sweep.projection_core import orth_base


@settings(max_examples=10, deadline=None)
@given(
    a=st.floats(min_value=-10.0, max_value=10.0),
    b=st.floats(min_value=-10.0, max_value=10.0),
)
def test_sweep_projection_linearity_invariant(a: float, b: float) -> None:
    """Mathematical invariant: SWeeP(a*X + b*Y) == a*SWeeP(X) + b*SWeeP(Y) in count_none."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 50.0, size=(4, 25))
    y = rng.uniform(0.0, 50.0, size=(4, 25))
    combined = a * x + b * y

    orth = orth_base(lin=25, col=5, seed=DEFAULT_SEED, dtype=np.float64)

    proj_x = sweep(
        x, orthbase=orth, bin_mode=False, norm="none", verbose=False, dtype=np.float64
    ).proj
    proj_y = sweep(
        y, orthbase=orth, bin_mode=False, norm="none", verbose=False, dtype=np.float64
    ).proj
    proj_comb = sweep(
        combined, orthbase=orth, bin_mode=False, norm="none", verbose=False, dtype=np.float64
    ).proj

    expected = a * proj_x + b * proj_y
    np.testing.assert_allclose(proj_comb, expected, rtol=1e-11, atol=1e-11)


@settings(max_examples=5, deadline=None)
@given(seed=st.integers(min_value=1, max_value=1000000))
def test_sparse_algebraic_identity_arbitrary_matrices(seed: int) -> None:
    """Proves the exactness of the sparse identity: X_tilde @ B == X_hat @ B - 1 @ (1.T @ B)."""
    rng = np.random.default_rng(seed)
    dense = rng.poisson(lam=0.5, size=(6, 20)).astype(np.float64)
    sparse = csr_array(dense)

    orth = orth_base(lin=20, col=4, seed=seed, dtype=np.float64)

    for norm in ("none", "log", "Neg", "logNeg"):
        p_sparse = sweep(
            sparse, orthbase=orth, bin_mode=False, norm=norm, verbose=False, dtype=np.float64
        ).proj
        p_dense = sweep(
            dense, orthbase=orth, bin_mode=False, norm=norm, verbose=False, dtype=np.float64
        ).proj
        np.testing.assert_allclose(p_sparse, p_dense, rtol=1e-11, atol=1e-11)
