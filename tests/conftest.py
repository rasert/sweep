from __future__ import annotations

import hashlib
from collections.abc import Generator
from pathlib import Path

import pytest

from tests.helpers.golden import (
    EXPECTED_MANIFEST_SHA256,
    get_fixtures_dir,
    get_manifest_path,
    load_golden_manifest,
    verify_golden_checksum,
)


@pytest.fixture(scope="session", autouse=True)
def guard_golden_fixtures_integrity() -> Generator[None]:
    """Ensures that golden files and manifest are never mutated by any test or sub-agent."""
    manifest_path = get_manifest_path()
    assert manifest_path.exists(), f"Manifest file missing: {manifest_path}"

    actual_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert actual_manifest_sha == EXPECTED_MANIFEST_SHA256, (
        f"Golden manifest.json has been tampered with! "
        f"Expected {EXPECTED_MANIFEST_SHA256}, got {actual_manifest_sha}"
    )

    manifest = load_golden_manifest()
    for filename in manifest.get("files", {}):
        assert verify_golden_checksum(filename), f"Checksum verification failed for {filename}"

    yield

    # Post-session check to confirm no mutations occurred during test execution
    post_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert post_manifest_sha == EXPECTED_MANIFEST_SHA256, (
        "Golden manifest.json was modified during pytest session!"
    )


@pytest.fixture
def golden_fixtures_dir() -> Path:
    return get_fixtures_dir()


@pytest.fixture
def examples_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "examples"


@pytest.fixture
def mito_fasta_dir(examples_dir: Path) -> Path:
    return examples_dir / "aaMitochondrial"


@pytest.fixture
def mito_tree_path(examples_dir: Path) -> Path:
    return examples_dir / "tree_Mitochondrial.tree"


@pytest.fixture
def mito_metadata_path(examples_dir: Path) -> Path:
    return examples_dir / "metadata_mitochondrial.csv"
