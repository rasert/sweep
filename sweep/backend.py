from __future__ import annotations

import logging
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from sweep.models import DeviceType

logger: Final[logging.Logger] = logging.getLogger(__name__)


def _detect_gpu() -> tuple[bool, str | None]:
    try:
        import cupy as cp

        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count > 0:
            props = cp.cuda.runtime.getDeviceProperties(0)
            raw_name = props["name"]
            dev_name = str(raw_name.decode("utf-8") if isinstance(raw_name, bytes) else raw_name)
            logger.info("NVIDIA GPU detected and activated: %s", dev_name)
            return True, dev_name
    except Exception as exc:
        logger.debug("GPU acceleration unavailable, fallback to CPU: %s", exc)
    return False, None


_CUPY_AVAILABLE, _GPU_DEVICE_NAME = _detect_gpu()


def is_gpu_available() -> bool:
    """Checks whether an NVIDIA GPU is detected and CuPy is operational."""
    return _CUPY_AVAILABLE


def get_gpu_device_name() -> str | None:
    """Returns the model name of the detected GPU, or None if unavailable."""
    return _GPU_DEVICE_NAME


def get_available_device(requested: DeviceType = "auto") -> DeviceType:
    """Resolves device choice between auto, cuda, and cpu."""
    if requested == "auto":
        return "cuda" if is_gpu_available() else "cpu"
    if requested == "cuda":
        if not is_gpu_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            return "cpu"
        return "cuda"
    return "cpu"


def sparse_dense_matmul(
    sp_mat: csr_array,
    dense_mat: NDArray[np.float64],
    device: DeviceType = "auto",
) -> NDArray[np.float64]:
    """Multiplies a CSR sparse matrix by a dense matrix in float64 precision."""
    target_device = get_available_device(device)

    if target_device == "cuda" and _CUPY_AVAILABLE:
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cp_sparse

            sp_gpu = cp_sparse.csr_matrix(
                (
                    cp.asarray(sp_mat.data, dtype=cp.float64),
                    cp.asarray(sp_mat.indices, dtype=cp.int32),
                    cp.asarray(sp_mat.indptr, dtype=cp.int32),
                ),
                shape=sp_mat.shape,
            )
            dense_gpu = cp.asarray(dense_mat, dtype=cp.float64)
            res_gpu = sp_gpu.dot(dense_gpu)
            return cp.asnumpy(res_gpu)
        except Exception as exc:
            logger.warning("GPU execution failed, falling back to CPU: %s", exc)

    return sp_mat.dot(dense_mat)


def dense_dense_matmul(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    device: DeviceType = "auto",
) -> NDArray[np.float64]:
    """Multiplies two dense matrices in float64 precision."""
    target_device = get_available_device(device)

    if target_device == "cuda" and _CUPY_AVAILABLE:
        try:
            import cupy as cp

            a_gpu = cp.asarray(a, dtype=cp.float64)
            b_gpu = cp.asarray(b, dtype=cp.float64)
            res_gpu = cp.matmul(a_gpu, b_gpu)
            return cp.asnumpy(res_gpu)
        except Exception as exc:
            logger.warning("GPU execution failed, falling back to CPU: %s", exc)

    return a @ b
