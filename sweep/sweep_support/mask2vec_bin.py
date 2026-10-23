#!/usr/bin/python3
# -*- coding: utf-8 -*-

from .seq2list import seq2list
from .aa2idx import aa2idx
from .ij2inds import ij2inds
import numpy as np


def mask2vec_bin(xseq, mask=[2, 1, 2], defSize=20):
    t = mask[0] + mask[1] + mask[2]
    slice = seq2list(xseq, int(t))
    xy = np.array(
        [aa2idx(slice[:, 0 : mask[0]], defSize), aa2idx(slice[:, t - (mask[2]) : t], defSize)]
    ).T
    xy = np.unique(xy, axis=0)
    lines = defSize ** mask[0]
    inds = ij2inds(xy, lines)
    cols = defSize ** mask[2]
    M = np.zeros((1, lines * cols), dtype=int)
    rows_size = defSize ** mask[0] * defSize ** mask[2]
    inds = inds[inds <= rows_size]
    M[0, inds - 1] = 1
    return M
