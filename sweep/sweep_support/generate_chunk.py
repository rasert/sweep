#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np


def generate_chunk(chunks, file_len):
    chunk_size = math.floor(file_len / chunks)
    chunk_indices = np.tile(chunk_size, (chunks, 2)) * np.array(
        [list(range(0, chunks)), list(range(1, chunks + 1))]
    ).T + np.concatenate((np.ones((chunks, 1)), np.zeros((chunks, 1))), axis=1)
    chunk_indices[-1, 1] = file_len
    return chunk_indices
