from __future__ import annotations

import logging
from typing import Any, Final

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
    dense_mat: NDArray[np.floating[Any]],
    device: DeviceType = "auto",
) -> NDArray[np.floating[Any]]:
    """Multiplies a CSR sparse matrix by a dense matrix, preserving precision (float32/float64)."""
    target_device = get_available_device(device)
    target_dtype = dense_mat.dtype

    if target_device == "cuda" and _CUPY_AVAILABLE:
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cp_sparse

            cp_dtype = cp.float32 if target_dtype == np.float32 else cp.float64
            sp_gpu = cp_sparse.csr_matrix(
                (
                    cp.asarray(sp_mat.data, dtype=cp_dtype),
                    cp.asarray(sp_mat.indices, dtype=cp.int32),
                    cp.asarray(sp_mat.indptr, dtype=cp.int32),
                ),
                shape=sp_mat.shape,
            )
            dense_gpu = cp.asarray(dense_mat, dtype=cp_dtype)
            res_gpu = sp_gpu.dot(dense_gpu)
            return cp.asnumpy(res_gpu)
        except Exception as exc:
            logger.warning("GPU execution failed, falling back to CPU: %s", exc)

    if sp_mat.dtype != target_dtype:
        sp_mat = sp_mat.astype(target_dtype)
    return sp_mat.dot(dense_mat)


def dense_dense_matmul(
    a: NDArray[np.floating[Any]],
    b: NDArray[np.floating[Any]],
    device: DeviceType = "auto",
) -> NDArray[np.floating[Any]]:
    """Multiplies two dense matrices, preserving precision (float32/float64)."""
    target_device = get_available_device(device)
    target_dtype = b.dtype

    if target_device == "cuda" and _CUPY_AVAILABLE:
        try:
            import cupy as cp

            cp_dtype = cp.float32 if target_dtype == np.float32 else cp.float64
            a_gpu = cp.asarray(a, dtype=cp_dtype)
            b_gpu = cp.asarray(b, dtype=cp_dtype)
            res_gpu = cp.matmul(a_gpu, b_gpu)
            return cp.asnumpy(res_gpu)
        except Exception as exc:
            logger.warning("GPU execution failed, falling back to CPU: %s", exc)

    if a.dtype != target_dtype:
        a = a.astype(target_dtype)
    return a @ b
