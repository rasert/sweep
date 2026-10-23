#!/usr/bin/python3
# -*- coding: utf-8 -*-

from .aa2int import aa2int
from .nt2int import nt2int
import numpy as np


def aa2idx(xseq, defSize):
    [n, m] = np.array(xseq).shape
    if defSize == 20:
        vls = (np.array(aa2int(xseq)) - 1).T
    elif defSize == 4:
        vls = (np.array(nt2int(xseq)) - 1).T
    pot = np.tile(range(0, m), (n, 1)).T
    t = np.tile(defSize, (m, n))
    mret = np.sum((t**pot) * vls, axis=0) + 1
    return mret
