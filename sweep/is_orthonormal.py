#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


def is_orthonormal(matrix):

    m, n = matrix.shape

    # Verifica se as colunas são vetores unitários
    for j in range(n):
        col = matrix[:, j]
        norm = np.linalg.norm(col)
        if abs(norm - 1) > 1e-10:
            return False

    # Verifica se as colunas são ortogonais entre si
    for j1 in range(n):
        for j2 in range(j1 + 1, n):
            dot_product = np.dot(matrix[:, j1], matrix[:, j2])
            if abs(dot_product) > 1e-10:
                return False

    # Se chegou até aqui, a matriz é ortonormal
    return True
