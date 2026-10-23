from __future__ import annotations

from pathlib import Path

import polars as pl

from sweep.models import PCCIResult, PMPGResult
from sweep.phylo import pcci, pmpg
from tests.helpers.golden import load_golden_df


def test_pcci_mitochondrial_tree(examples_dir: Path) -> None:
    tree_path = examples_dir / "tree_Mitochondrial.tree"
    meta_path = examples_dir / "metadata_mitochondrial.csv"

    res = pcci(tree_path, meta_path)

    assert isinstance(res, PCCIResult)
    assert isinstance(res.tab, pl.DataFrame)
    assert res.mean == 1.0

    golden_df = load_golden_df("pcci_mitochondrial.csv")
    assert res.tab["taxa"].to_list() == golden_df["taxa"].to_list()
    actual_costs = res.tab["cost"].cast(pl.Float64).to_list()
    expected_costs = golden_df["cost"].cast(pl.Float64).to_list()
    assert actual_costs == expected_costs


def test_pmpg_mitochondrial_tree(examples_dir: Path) -> None:
    tree_path = examples_dir / "tree_Mitochondrial.tree"
    meta_path = examples_dir / "metadata_mitochondrial.csv"

    res = pmpg(tree_path, meta_path)

    assert isinstance(res, PMPGResult)
    assert isinstance(res.tab, pl.DataFrame)
    assert res.perc_mono == 100.0
    assert res.perc_para == 0.0
    assert res.metric == 100.0

    golden_df = load_golden_df("pmpg_mitochondrial.csv")
    assert res.tab["taxa"].to_list() == golden_df["taxa"].to_list()
    expected_mono = [v == "TRUE" or v is True for v in golden_df["mono"]]
    expected_para = [v == "TRUE" or v is True for v in golden_df["para"]]
    assert res.tab["mono"].to_list() == expected_mono
    assert res.tab["para"].to_list() == expected_para
