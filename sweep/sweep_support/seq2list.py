#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def seq2list(sdna, q):
    sdna = np.array(list(sdna))
    n = sdna.shape[0]
    if n < q:
        return np.array([])
    b = np.zeros((n - q + 1, q), dtype=sdna.dtype)
    for i in range(q):
        b[:, i] = sdna[i : n - q + i + 1]
    return b
