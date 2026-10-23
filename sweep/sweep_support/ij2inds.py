#!/usr/bin/python3
# -*- coding: utf-8 -*-


def ij2inds(ijs, tmcol):
    return ((ijs[:, 1] - 1) * tmcol) + (ijs[:, 0])
