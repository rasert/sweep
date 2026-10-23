#!/usr/bin/python3
# -*- coding: utf-8 -*-

from .sweep_support.orth import orth
import warnings
import numpy as np


def orthbase(lin, col, seed=None, dtype=np.float32):
    warnings.warn(
        "'orthbase' is deprecated and will be removed in a future release; use 'orth_base' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    """
    Generate an orthogonal basis matrix.

    Args:
        - lin (int): Number of rows in the matrix.
        - col (int): Number of columns in the matrix.
        - seed (int, optional): Seed for the random number generator.
        - dtype (data-type, optional): Desired data-type for the array,
          default is np.float32.

    Returns:
        - ndarray: Orthogonal basis matrix.
    """
    if seed is not None:
        np.random.seed(seed)

    if lin != col:
        Ro = orth(np.random.rand(lin, col + 1).astype(dtype))
        mret = Ro[:, 1:]
    else:
        mret = orth(np.random.rand(lin, col).astype(dtype))

    return np.array(mret, dtype=dtype)
