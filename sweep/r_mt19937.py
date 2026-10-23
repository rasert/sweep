from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from sweep.models import DEFAULT_SEED

_N: Final[int] = 624
_M: Final[int] = 397
_MATRIX_A: Final[int] = 0x9908B0DF
_UPPER_MASK: Final[int] = 0x80000000
_LOWER_MASK: Final[int] = 0x7FFFFFFF
_TEMPERING_MASK_B: Final[int] = 0x9D2C5680
_TEMPERING_MASK_C: Final[int] = 0xEFC60000
_MAG01: Final[tuple[int, int]] = (0x0, _MATRIX_A)
_SCALE: Final[float] = 2.3283064365386963e-10  # 1.0 / 2^32


def _fixup(x: float) -> float:
    """R's fixup function to ensure exact 0.0 and 1.0 are never returned."""
    if x <= 0.0:
        return 0.5 * _SCALE
    if (1.0 - x) <= 0.0:
        return 1.0 - 0.5 * _SCALE
    return x


class R_MT19937:
    """Exact implementation of GNU R's Mersenne Twister PRNG.

    Reproduces stats::runif() bit-for-bit to full IEEE 754 double precision.
    """

    __slots__ = ("_mt", "_mti")

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self._mt = np.zeros(_N, dtype=np.uint32)
        self._mti = _N
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """Initializes generator state following R's 51-iteration LCG warmup."""
        current_seed = seed & 0xFFFFFFFF
        for _ in range(51):
            current_seed = (69069 * current_seed + 1) & 0xFFFFFFFF

        for i in range(_N):
            current_seed = (69069 * current_seed + 1) & 0xFFFFFFFF
            self._mt[i] = current_seed

        self._mti = _N

    def _twist(self) -> None:
        """Generates 624 words at once via Mersenne Twister recurrence."""
        mt = self._mt
        for kk in range(_N - _M):
            y = int((mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK))
            mt[kk] = int(mt[kk + _M]) ^ (y >> 1) ^ _MAG01[y & 0x1]

        for kk in range(_N - _M, _N - 1):
            y = int((mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK))
            mt[kk] = int(mt[kk + (_M - _N)]) ^ (y >> 1) ^ _MAG01[y & 0x1]

        y = int((mt[_N - 1] & _UPPER_MASK) | (mt[0] & _LOWER_MASK))
        mt[_N - 1] = int(mt[_M - 1]) ^ (y >> 1) ^ _MAG01[y & 0x1]
        self._mti = 0

    def next_double(self) -> float:
        """Generates a single uniform deviate matching R's unif_rand()."""
        if self._mti >= _N:
            self._twist()

        y = int(self._mt[self._mti])
        self._mti += 1

        y ^= y >> 11
        y ^= (y << 7) & _TEMPERING_MASK_B
        y ^= (y << 15) & _TEMPERING_MASK_C
        y ^= y >> 18

        raw = float(y) * _SCALE
        return _fixup(raw)

    def runif(self, n: int) -> NDArray[np.float64]:
        """Generates n uniform random numbers matching R's stats::runif(n)."""
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self.next_double()
        return out
