#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def orth(A):
    Q, S, Vt = np.linalg.svd(A, full_matrices=False)
    tol = max(A.shape) * S[0] * np.finfo("double").eps
    r = sum(S > tol)
    Q = Q[:, 0:r]
    return Q
