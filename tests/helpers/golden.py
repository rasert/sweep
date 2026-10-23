from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import polars as pl
from numpy.typing import NDArray
from scipy.io import mmread
from scipy.sparse import csr_array

EXPECTED_MANIFEST_SHA256: str = "8ec2c6f6c8c41a0a1238efdbd229a286921056b60137b892ea6eb0a4ea62223d"


def get_fixtures_dir() -> Path:
    """Returns the absolute path to tests/fixtures/golden."""
    return Path(__file__).resolve().parent.parent / "fixtures" / "golden"


def get_manifest_path() -> Path:
    """Returns the absolute path to tests/fixtures/golden/manifest.json."""
    return get_fixtures_dir() / "manifest.json"


def load_golden_manifest() -> dict[str, Any]:
    """Loads and returns the golden manifest dictionary."""
    mpath = get_manifest_path()
    if not mpath.exists():
        msg = f"Manifest not found at: {mpath}"
        raise FileNotFoundError(msg)
    data = orjson.loads(mpath.read_bytes())
    if not isinstance(data, dict):
        msg = "Manifest is not a valid JSON dictionary."
        raise TypeError(msg)
    return data


def verify_golden_checksum(filename: str) -> bool:
    """Verifies SHA-256 of the given golden file against manifest.json."""
    filepath = get_fixtures_dir() / filename
    if not filepath.exists():
        msg = f"Golden file not found: {filepath}"
        raise FileNotFoundError(msg)

    manifest = load_golden_manifest()
    files_entry = manifest.get("files", {})
    entry = files_entry.get(filename)
    if entry is None:
        msg = f"Filename not registered in manifest: {filename}"
        raise KeyError(msg)

    actual_sha = hashlib.sha256(filepath.read_bytes()).hexdigest()
    expected_sha = entry["sha256"]
    if actual_sha != expected_sha:
        msg = f"Checksum mismatch for {filename}: expected {expected_sha}, got {actual_sha}"
        raise ValueError(msg)
    return True


def load_golden_csv(filename: str) -> NDArray[np.float64]:
    """Loads a golden CSV as a float64 numpy array, dropping the row names."""
    verify_golden_checksum(filename)
    filepath = get_fixtures_dir() / filename
    df = pl.read_csv(filepath)
    # The first column is row names (sample/feature identifiers)
    data_df = df.select(pl.all().exclude(df.columns[0]))
    return data_df.to_numpy().astype(np.float64)


def load_golden_df(filename: str) -> pl.DataFrame:
    """Loads a golden CSV as a Polars DataFrame."""
    verify_golden_checksum(filename)
    filepath = get_fixtures_dir() / filename
    return pl.read_csv(filepath)


def load_golden_mtx(filename: str) -> csr_array:
    """Loads a golden MatrixMarket file as a SciPy CSR array."""
    verify_golden_checksum(filename)
    filepath = get_fixtures_dir() / filename
    mat = mmread(filepath)
    return csr_array(mat)
