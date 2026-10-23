#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


def nt2int(nt_list):
    map = {
        "a": 1,
        "b": 11,
        "c": 2,
        "d": 12,
        "e": 0,
        "f": 0,
        "g": 3,
        "h": 13,
        "i": 0,
        "j": 0,
        "k": 7,
        "l": 0,
        "m": 8,
        "n": 15,
        "o": 0,
        "p": 0,
        "q": 0,
        "r": 5,
        "s": 9,
        "t": 4,
        "u": 4,
        "v": 14,
        "w": 10,
        "x": 15,
        "y": 6,
        "z": 0,
        "*": 0,
        "-": 16,
        "?": 0,
    }
    nt_list = np.char.lower(nt_list)
    map_func = np.vectorize(map.get)
    nt_list = map_func(nt_list)
    return nt_list
